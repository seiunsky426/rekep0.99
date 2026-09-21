// Small C ABI: OMPL search and MoveIt self-collision, no ROS motion interfaces.
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/MotionValidator.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <moveit/planning_scene/planning_scene.h>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace ob = ompl::base;
namespace og = ompl::geometric;
using Check = int (*)(const double*);

class EdgeCheck : public ob::MotionValidator {
 public:
  EdgeCheck(const ob::SpaceInformationPtr& si, Check check)
      : ob::MotionValidator(si), check_(check) {}
  bool checkMotion(const ob::State* a, const ob::State* b) const override {
    std::pair<ob::State*, double> last(nullptr, 0.);
    return checkMotion(a, b, last);
  }
  bool checkMotion(const ob::State* a, const ob::State* b,
                   std::pair<ob::State*, double>& last) const override {
    auto x = a->as<ob::RealVectorStateSpace::StateType>()->values;
    auto y = b->as<ob::RealVectorStateSpace::StateType>()->values;
    double travel = 0.;
    for (int j=0;j<6;++j) travel += std::abs(x[j]-y[j]);
    int n = std::max(1, static_cast<int>(std::ceil(travel/.005)));
    double q[6];
    for (int i=0;i<=n;++i) {
      for (int j=0;j<6;++j) q[j] = x[j]+(y[j]-x[j])*i/n;
      if (!check_(q)) {
        last.second = std::max(0., (i-1.)/n);
        if (last.first) si_->getStateSpace()->interpolate(a,b,last.second,last.first);
        return false;
      }
    }
    return true;
  }
 private:
  Check check_;
};

class SeededRRT : public og::RRTConnect {
 public:
  SeededRRT(const ob::SpaceInformationPtr& si, unsigned seed) : og::RRTConnect(si) {
    rng_.setLocalSeed(seed);
  }
};
class SeededSampler : public ob::RealVectorStateSampler {
 public:
  SeededSampler(const ob::StateSpace* space, unsigned seed)
      : ob::RealVectorStateSampler(space) { rng_.setLocalSeed(seed); }
};

extern "C" int rekpiper_rrt(const double* lower, const double* upper,
                            const double* start, const double* goal, Check check,
                            double seconds, unsigned seed, double* output, int capacity) {
  try {
    if (!check || !output || capacity < 2 || seconds <= 0.) return -1;
    auto space = std::make_shared<ob::RealVectorStateSpace>(6);
    ob::RealVectorBounds bounds(6);
    for (int i=0;i<6;++i) { bounds.setLow(i,lower[i]); bounds.setHigh(i,upper[i]); }
    space->setBounds(bounds);
    space->setStateSamplerAllocator([seed](const ob::StateSpace* s) {
      return std::make_shared<SeededSampler>(s,seed);
    });
    og::SimpleSetup setup(space);
    setup.setStateValidityChecker([check](const ob::State* state) {
      return check(state->as<ob::RealVectorStateSpace::StateType>()->values) != 0;
    });
    setup.getSpaceInformation()->setMotionValidator(
        std::make_shared<EdgeCheck>(setup.getSpaceInformation(),check));
    ob::ScopedState<> a(space), b(space);
    for (int i=0;i<6;++i) { a[i]=start[i]; b[i]=goal[i]; }
    setup.setStartAndGoalStates(a,b);
    auto planner=std::make_shared<SeededRRT>(setup.getSpaceInformation(),seed);
    planner->setRange(.20);
    setup.setPlanner(planner);
    setup.solve(seconds);
    if (!setup.haveExactSolutionPath()) return 0;
    const auto& states=setup.getSolutionPath().getStates();
    if (static_cast<int>(states.size())>capacity) return -2;
    for (size_t i=0;i<states.size();++i)
      std::copy_n(states[i]->as<ob::RealVectorStateSpace::StateType>()->values,6,output+6*i);
    return static_cast<int>(states.size());
  } catch (...) { return -3; }
}

struct CollisionModel {
  moveit::core::RobotModelPtr model;
  planning_scene::PlanningScenePtr scene;
};

extern "C" void* rekpiper_collision_create(const char* xml) {
  try {
    auto urdf=urdf::parseURDF(xml);
    if (!urdf) return nullptr;
    auto semantic=std::make_shared<srdf::Model>();
    std::string srdf="<robot name=\""+urdf->getName()+"\">";
    // Only directly adjacent links may touch by construction.
    for (const auto& item : urdf->joints_) {
      auto j=item.second;
      srdf+="<disable_collisions link1=\""+j->parent_link_name+
            "\" link2=\""+j->child_link_name+"\" reason=\"Adjacent\"/>";
    }
    srdf+="</robot>";
    if (!semantic->initString(*urdf,srdf)) return nullptr;
    auto c=std::unique_ptr<CollisionModel>(new CollisionModel);
    c->model=std::make_shared<moveit::core::RobotModel>(urdf,semantic);
    for (const auto& item : urdf->links_) {
      if (!item.second->collision && item.second->collision_array.empty()) continue;
      const auto* link=c->model->getLinkModel(item.first);
      if (!link || link->getShapes().empty()) return nullptr;
    }
    c->scene=std::make_shared<planning_scene::PlanningScene>(c->model);
    return c.release();
  } catch (...) { return nullptr; }
}
extern "C" int rekpiper_self_clear(void* ptr, const double* joints, double opening) {
  try {
    auto* c=static_cast<CollisionModel*>(ptr);
    if (!c) return 0;
    moveit::core::RobotState state(c->model);
    state.setToDefaultValues();
    if (!std::isfinite(opening) || opening < 0. || opening > .070) return 0;
    if (c->model->hasJointModel("joint7")) state.setVariablePosition("joint7",opening/2.);
    if (c->model->hasJointModel("joint8")) state.setVariablePosition("joint8",-opening/2.);
    for (int i=0;i<6;++i) {
      std::string name="joint"+std::to_string(i+1);
      if (!c->model->hasJointModel(name)) return 0;
      state.setVariablePosition(name,joints[i]);
    }
    state.update();
    collision_detection::CollisionRequest request;
    collision_detection::CollisionResult result;
    c->scene->checkSelfCollision(request,result,state);
    return result.collision ? 0 : 1;
  } catch (...) { return 0; }
}
extern "C" void rekpiper_collision_destroy(void* ptr) {
  delete static_cast<CollisionModel*>(ptr);
}
