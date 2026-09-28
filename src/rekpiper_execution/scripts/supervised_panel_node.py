#!/usr/bin/env python3
"""Single-window supervised pick/place panel; all ROS calls run off the Qt thread."""
import json
import threading
import time

import actionlib
from cv_bridge import CvBridge
from PyQt5 import QtCore, QtGui, QtWidgets
import rospy
from rviz import bindings as rviz
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rekpiper_msgs.msg import (PlanSupervisedStageAction, PlanSupervisedStageGoal,
    ExecuteSupervisedStageAction, ExecuteSupervisedStageGoal)
from rekpiper_msgs.srv import SupervisedCommand

NS = '/rekpiper/supervised'


class Picture(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(str, int, int)

    def __init__(self, name):
        super().__init__('等待'+name+'图像')
        self.name = name
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(340, 210)
        self.setStyleSheet('background:#222831;color:#dddddd')
        self.image = None
        self.stamp = 0.
        self.calibration_click = False

    def show_image(self, image, stamp):
        self.image = image.copy()
        self.stamp = stamp
        h, w = image.shape[:2]
        frame = QtGui.QImage(image.data, w, h, image.strides[0], QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(frame.copy()).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.setToolTip('%s 采集时间 %.3f' % (self.name, stamp))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image is not None:
            self.show_image(self.image, self.stamp)

    def mousePressEvent(self, event):
        if not self.calibration_click or self.image is None:
            return
        h, w = self.image.shape[:2]
        scale = min(self.width()/w, self.height()/h)
        dw, dh = w*scale, h*scale
        x = (event.x()-(self.width()-dw)/2)/scale
        y = (event.y()-(self.height()-dh)/2)/scale
        if 0 <= x < w and 0 <= y < h:
            self.clicked.emit(self.name, int(x), int(y))


class Signals(QtCore.QObject):
    status = QtCore.pyqtSignal(dict)
    perception = QtCore.pyqtSignal(dict)
    image = QtCore.pyqtSignal(str, object, float)
    finished = QtCore.pyqtSignal(str, bool, str)
    phase = QtCore.pyqtSignal(str)


class Panel(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ReKpiper 人工监督抓放工作台')
        self.resize(1800, 1050)
        self.signals = Signals()
        self.signals.status.connect(self.apply_status)
        self.signals.perception.connect(self.apply_perception)
        self.signals.image.connect(self.apply_image)
        self.signals.finished.connect(self.finished)
        self.signals.phase.connect(self.message)
        self.bridge = CvBridge()
        self.status = {}
        self.perception = {}
        self.busy = False
        self.snapshot = False
        self.calibration_frozen = False
        self.pixels = {}
        self.images = {}
        self.buttons = {}
        self._built = False
        self.plan_client = actionlib.SimpleActionClient(NS+'/plan', PlanSupervisedStageAction)
        self.execute_client = actionlib.SimpleActionClient(NS+'/execute', ExecuteSupervisedStageAction)
        self.heartbeat = rospy.Publisher(NS+'/heartbeat', String, queue_size=1)
        rospy.Subscriber(NS+'/status', String, self.status_cb, queue_size=1)
        rospy.Subscriber('/rekpiper/perception/task_status', String, self.perception_cb, queue_size=1)
        for name in ('rs1','rs3'):
            rospy.Subscriber('/'+name+'/color/image_raw', Image,
                lambda msg, n=name: self.image_cb(n,msg), queue_size=1)
            rospy.Subscriber(NS+'/calibration/'+name, Image,
                lambda msg, n=name: self.image_cb('calibration_'+n,msg), queue_size=1)
        rospy.Subscriber('/rekpiper/perception/candidate_image', Image,
            lambda msg: self.image_cb('candidate',msg), queue_size=1)
        self.build()
        self._built = True
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

    def build(self):
        central=QtWidgets.QWidget(); self.setCentralWidget(central)
        outer=QtWidgets.QHBoxLayout(central)
        left=QtWidgets.QScrollArea(); left.setWidgetResizable(True); left.setMinimumWidth(540)
        pane=QtWidgets.QWidget(); left.setWidget(pane); form=QtWidgets.QVBoxLayout(pane)
        outer.addWidget(left, 1)
        right=QtWidgets.QWidget(); rv=QtWidgets.QVBoxLayout(right); outer.addWidget(right, 3)
        views=QtWidgets.QHBoxLayout();rv.addLayout(views,2)
        self.rs1=Picture('rs1');self.rs3=Picture('rs3')
        for pic in (self.rs1,self.rs3):
            pic.clicked.connect(self.pick)
            box=QtWidgets.QVBoxLayout();box.addWidget(QtWidgets.QLabel(pic.name.upper()));box.addWidget(pic)
            views.addLayout(box)
        self.rviz=rviz.VisualizationFrame()
        self.rviz.setSplashPath('')
        self.rviz.initialize()
        config=rviz.Config()
        reader=rviz.YamlConfigReader()
        reader.readFile(config,rospy.get_param('~rviz_config'))
        self.rviz.load(config)
        self.rviz.setMenuBar(None)
        self.rviz.setStatusBar(None)
        self.rviz.setHideButtonVisibility(False)
        rv.addWidget(self.rviz,4)
        self.picture_state=QtWidgets.QLabel('图像时间与状态等待中；RViz 坐标系 base_link')
        rv.addWidget(self.picture_state)
        self.add_title(form,'0 机械臂')
        row=QtWidgets.QHBoxLayout();form.addLayout(row)
        for label,cmd in [('连接 CAN','connect'),('使能','enable'),('停止','stop'),('失能','disable')]:
            self.add_button(row,label,lambda _,c=cmd:self.command(c),cmd)
        self.add_title(form,'1 相机')
        self.add_button(form,'启动相机',lambda:self.command('cameras'),'cameras')
        self.add_title(form,'2 关键点与固定约束')
        self.instruction=QtWidgets.QLineEdit('把蓝色方块放到黄色圆盘上')
        self.instruction.setReadOnly(True)
        form.addWidget(self.instruction)
        self.task_debounce=QtCore.QTimer(self);self.task_debounce.setSingleShot(True)
        row=QtWidgets.QHBoxLayout();form.addLayout(row)
        self.add_button(row,'SAM＋DINOv2 生成关键点',lambda:self.command('keypoints'),'keypoints')
        self.toggle=QtWidgets.QCheckBox('显示编号快照');self.toggle.toggled.connect(self.toggle_snapshot);row.addWidget(self.toggle)
        self.add_button(form,'VLM 选择蓝色方块／黄色圆盘并填入固定约束',lambda:self.command('task'),'task')
        self.add_title(form,'3 执行约束（逐段预览后执行）')
        self.add_button(form,'执行前坐标复核（展开／收起）',
                        lambda:self.geometry_box.setVisible(not self.geometry_box.isVisible()),'calibration_toggle')
        self.geometry_box=QtWidgets.QWidget()
        geometry_form=QtWidgets.QVBoxLayout(self.geometry_box)
        row=QtWidgets.QHBoxLayout();geometry_form.addLayout(row)
        self.add_button(row,'冻结复核图',lambda:self.command('calibration_freeze'),'calibration_freeze')
        coords=QtWidgets.QHBoxLayout();geometry_form.addLayout(coords)
        self.xyz=[]
        for axis in 'XYZ':
            field=QtWidgets.QDoubleSpinBox();field.setRange(-2,2);field.setDecimals(4)
            field.setSingleStep(.005);field.setSuffix(' m');coords.addWidget(QtWidgets.QLabel(axis));coords.addWidget(field)
            self.xyz.append(field)
        self.cal_label=QtWidgets.QLabel('执行相机关键点目标前，在冻结的 RS1、RS3 图像各点一次，录入独立测量的基座坐标')
        self.cal_label.setWordWrap(True);geometry_form.addWidget(self.cal_label)
        row=QtWidgets.QHBoxLayout();geometry_form.addLayout(row)
        self.add_button(row,'加入复核点',self.add_point,'calibration_point')
        self.add_button(row,'清除点',lambda:self.command('calibration_clear'),'calibration_clear')
        self.add_button(row,'复核坐标',lambda:self.command('calibration_verify'),'calibration_verify')
        form.addWidget(self.geometry_box);self.geometry_box.setVisible(False)
        self.stage_cards=[]
        for title in ('1 预抓取','2 抓取','3 抬升并运输','4 下放释放'):
            label=QtWidgets.QLabel(title);label.setStyleSheet('padding:6px;border:1px solid #888')
            form.addWidget(label);self.stage_cards.append(label)
        row=QtWidgets.QHBoxLayout();form.addLayout(row)
        self.velocity=QtWidgets.QDoubleSpinBox();self.velocity.setRange(.001,.2);self.velocity.setDecimals(3)
        self.velocity.setValue(.05);self.velocity.setSuffix(' rad/s');row.addWidget(self.velocity)
        self.driver=QtWidgets.QSpinBox();self.driver.setRange(1,20);self.driver.setValue(5)
        self.driver.setSuffix(' %');row.addWidget(self.driver)
        self.add_button(row,'设置速度',self.set_speed,'speed')
        row=QtWidgets.QHBoxLayout();form.addLayout(row)
        self.add_button(row,'生成本段预览',self.plan,'plan')
        self.add_button(row,'执行本段',self.execute,'execute')
        row=QtWidgets.QHBoxLayout();form.addLayout(row)
        self.add_button(row,'已夹稳',lambda:self.command('grasp',{'confirmed':True}),'grasp_yes')
        self.add_button(row,'未夹稳',lambda:self.command('grasp',{'confirmed':False}),'grasp_no')
        self.add_button(row,'清除留影',lambda:self.command('clear_trace'),'clear_trace')
        self.detail=QtWidgets.QPlainTextEdit();self.detail.setReadOnly(True);self.detail.setMinimumHeight(210)
        form.addWidget(self.detail)
        form.addStretch(1)
        self.statusBar().showMessage('等待会话服务')

    def add_title(self,form,title):
        label=QtWidgets.QLabel(title);label.setStyleSheet('font-size:16px;font-weight:bold;margin-top:12px')
        form.addWidget(label)

    def add_button(self,layout,label,callback,key):
        button=QtWidgets.QPushButton(label);button.clicked.connect(callback)
        layout.addWidget(button);self.buttons[key]=button

    def status_cb(self,msg):
        try:self.signals.status.emit(json.loads(msg.data))
        except (ValueError,TypeError):pass

    def perception_cb(self,msg):
        try:self.signals.perception.emit(json.loads(msg.data))
        except (ValueError,TypeError):pass

    def apply_perception(self,status):
        if status != self.perception:
            self.perception=status
            if 'failed' in status.get('detail',''):
                self.message('关键点生成受阻：'+status['detail'])

    def image_cb(self,name,msg):
        try:
            image=self.bridge.imgmsg_to_cv2(msg,'rgb8').copy()
            self.signals.image.emit(name,image,msg.header.stamp.to_sec())
        except Exception:pass

    def apply_image(self,name,image,stamp):
        self.images[name]=(image,stamp)
        if name in ('rs1','rs3') and not self.calibration_frozen and not (name=='rs1' and self.snapshot):
            getattr(self,name).show_image(image,stamp)
        if name=='candidate' and self.snapshot and not self.calibration_frozen:
            self.rs1.show_image(image,stamp)
        if name.startswith('calibration_') and self.calibration_frozen:
            getattr(self,name[len('calibration_'):]).show_image(image,stamp)

    def toggle_snapshot(self,value):
        self.snapshot=value
        self.refresh_picture()

    def refresh_picture(self):
        if self.calibration_frozen:return
        for name in ('rs1','rs3'):
            source='candidate' if name=='rs1' and self.snapshot else name
            if source in self.images:
                getattr(self,name).show_image(*self.images[source])

    def pick(self,name,x,y):
        self.pixels[name]=[x,y]
        self.cal_label.setText('冻结图像点选：RS1 %s；RS3 %s' %
            (self.pixels.get('rs1','待点'),self.pixels.get('rs3','待点')))

    def add_point(self):
        if set(self.pixels)!={'rs1','rs3'}:
            self.message('须在两幅冻结图像上各点一个对应点');return
        args={'base':[item.value() for item in self.xyz],**self.pixels}
        self.command('calibration_point',args)
        self.pixels={}

    def set_speed(self):
        self.command('speed',{'velocity':self.velocity.value(),'driver_percent':self.driver.value()})

    def command(self,name,args=None):
        if self.busy and name!='stop':return
        if name=='stop':
            self.plan_client.cancel_all_goals();self.execute_client.cancel_all_goals()
        else:self.busy=True
        def run():
            try:
                rospy.wait_for_service(NS+'/command',timeout=3)
                response=rospy.ServiceProxy(NS+'/command',SupervisedCommand)(name,json.dumps(args or {}))
                self.signals.finished.emit(name,response.success,response.status)
            except Exception as exc:self.signals.finished.emit(name,False,str(exc))
        threading.Thread(target=run,daemon=True).start()

    def action(self,name,client,goal):
        if self.busy:return
        self.busy=True
        def run():
            try:
                if not client.wait_for_server(rospy.Duration(3.)):
                    raise RuntimeError('阶段服务未启动')
                client.send_goal(goal,feedback_cb=lambda feedback:self.signals.phase.emit(feedback.phase))
                client.wait_for_result()
                result=client.get_result()
                self.signals.finished.emit(name,bool(result and result.success),
                    result.status if result else '动作取消')
            except Exception as exc:self.signals.finished.emit(name,False,str(exc))
        threading.Thread(target=run,daemon=True).start()

    def plan(self):
        s=self.status
        self.action('plan',self.plan_client,PlanSupervisedStageGoal(s.get('session_id',''),s.get('stage',0)))

    def execute(self):
        s=self.status;preview=s.get('preview') or {}
        self.action('execute',self.execute_client,ExecuteSupervisedStageGoal(
            s.get('session_id',''),s.get('stage',0),preview.get('id','')))

    def finished(self,name,success,message):
        if name!='stop':self.busy=False
        if name=='calibration_freeze' and success:
            self.calibration_frozen=True
            self.pixels={}
            for camera in ('rs1','rs3'):
                key='calibration_'+camera
                if key in self.images:getattr(self,camera).show_image(*self.images[key])
            for picture in (self.rs1,self.rs3):picture.calibration_click=True
        if name in ('calibration_verify','calibration_clear') and success:
            self.calibration_frozen=False
            for picture in (self.rs1,self.rs3):picture.calibration_click=False
            self.refresh_picture()
        self.message(('%s：'%('完成' if success else '失败'))+message)

    def message(self,message):
        self.statusBar().showMessage(message)

    def apply_status(self,status):
        self.status=status
        if not self._built:
            return
        state=status.get('state','IDLE');stage=int(status.get('stage',1))
        preview=status.get('preview')
        for i,card in enumerate(self.stage_cards,1):
            suffix=' ← 当前' if i==stage else (' ✓' if i<stage else '')
            card.setText('%d %s%s' % (i,status.get('stage_names',['预抓取','抓取','抬升并运输','下放释放'])[i-1],suffix))
            card.setStyleSheet('padding:6px;border:1px solid %s' % ('#38a169' if i==stage else '#888'))
        lines=['会话 %s 阶段 %d 状态 %s'%(status.get('session_id','')[:8],stage,state),
            status.get('reason',''), '外参复核 %s；复核点 %d；关节反馈 %.2f s' %
            ('通过' if status.get('calibration_valid') else '待完成',len(status.get('calibration_rows',[])),status.get('feedback_age_s',999)),
            '关键点 %s；选中 %s'%(len(status.get('keypoints',[])),
                [k['id'] for k in status.get('selection') or []])]
        if self.perception.get('task_serial',0):
            lines.append('关键点生成: %s；%s' % (
                self.perception.get('state',''),self.perception.get('detail','')))
        for key in status.get('selection') or []:
            lines.append('K%s 三维基座坐标 %s m' % (key.get('id'),key.get('position')))
        if preview and not status.get('calibration_valid'):
            lines.append('预览仅供查看：执行仍需外参现场复核')
        if preview:
            d=preview.get('detail',{})
            lines.extend(['预览 '+preview['id'][:8], '候选 '+str(d.get('source','')),
                '目标 TCP '+str(d.get('goal_matrix','')), '关节角 '+str(d.get('goal_joints','')),
                '时长 '+str(d.get('duration_s','')), '检查 '+str(d.get('audit','')),
                '淘汰原因 '+str(d.get('rejections',''))])
        if status.get('arm_status') is not None:
            lines.append('控制器: arm_status=%s teach_status=%s ctrl_mode=%s' %
                         (status['arm_status'],status['teach_status'],status['ctrl_mode']))
        if status.get('teach_status')==1:
            lines.append('控制器报告示教中：仅在绿灯常亮且机械臂已托稳时单击绿键退出；绿灯不亮时先核对控制器停机状态，勿操作绿键或双击')
        if status.get('arm_status')==1:
            lines.append('控制器报告停机（arm_status=1）：不能使能；恢复操作可能使机械臂下落')
        if status.get('raw_joints') is not None:
            lines.append('实测关节角 %s；反馈 %.2f s' %
                         (status['raw_joints'],status.get('raw_feedback_age_s',999)))
        constraints=status.get('constraints')
        if constraints:
            lines.append('固定约束: 蓝色方块 K%s %s m → 黄色圆盘 K%s %s m' % (
                constraints['blue_cube']['keypoint_id'],constraints['blue_cube']['position_m'],
                constraints['yellow_disk']['keypoint_id'],constraints['yellow_disk']['position_m']))
        stamp=status.get('tracking_stamp',0.)
        if stamp:
            lines.append('关键点跟踪: 有效 %d 个，数据距今 %.2f s；RViz 显示轨迹' % (
                status.get('tracking_valid',0),max(0.,time.time()-stamp)))

        if status.get('joint_validation_error'):
            lines.append('关节模型检查: '+status['joint_validation_error'])
        if status.get('camera_error'):lines.append('相机状态 '+status['camera_error'])
        if status.get('warning'):lines.append('警告 '+status['warning'])
        self.detail.setPlainText('\n'.join(lines))
        pictures=[]
        for name in ('rs1','rs3'):
            source=('calibration_'+name if self.calibration_frozen else
                    ('candidate' if name=='rs1' and self.snapshot else name))
            stamp=self.images.get(source,(None,0))[1]
            kind='冻结复核' if self.calibration_frozen else ('编号快照' if source=='candidate' else '实时')
            age=time.time()-stamp if stamp else 999.
            pictures.append('%s %s %.3f，距今 %.2f s，%s'%(name.upper(),kind,stamp,age,
                '新鲜' if age < .5 else '过期'))
        camera_error=status.get('camera_error','')
        self.picture_state.setText(('相机状态 '+camera_error+'；' if camera_error else '')+
            '；'.join(pictures)+'；图像坐标 rs1/rs3_color_optical_frame；RViz base_link')
        available=not self.busy
        self.buttons['plan'].setEnabled(available and stage<=4 and state=='IDLE' and bool(status.get('selection')))
        self.buttons['execute'].setEnabled(available and state=='PREVIEW_READY' and bool(preview)
            and bool(status.get('calibration_valid')) and not self.task_debounce.isActive()
            and self.instruction.text()==status.get('instruction',''))
        for name in ('grasp_yes','grasp_no'):
            self.buttons[name].setEnabled(available and state=='WAITING_GRASP')
        self.buttons['enable'].setEnabled(available and status.get('connected',False))
        self.buttons['connect'].setEnabled(available and not status.get('connected',False))
        self.buttons['stop'].setEnabled(True)

    def tick(self):
        token=self.status.get('control_id')
        if token:self.heartbeat.publish(String(token))

    def closeEvent(self,event):
        self.plan_client.cancel_all_goals();self.execute_client.cancel_all_goals()
        self.command('stop')
        super().closeEvent(event)


if __name__=='__main__':
    import sys
    rospy.init_node('panel',disable_signals=True)
    app=QtWidgets.QApplication(sys.argv)
    panel=Panel();panel.show()
    sys.exit(app.exec_())
