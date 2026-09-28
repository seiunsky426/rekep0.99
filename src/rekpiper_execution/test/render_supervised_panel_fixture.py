#!/usr/bin/env /usr/bin/python3
"""Render the workbench with recorded-shaped fake images and no ROS devices."""
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt5 import QtWidgets
import cv2
import time

root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('supervised_panel',
    root/'src/rekpiper_execution/scripts/supervised_panel_node.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PreviewFrame(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('background:#263238;border:2px solid #5c7382')
        layout=QtWidgets.QVBoxLayout(self)
        text=QtWidgets.QLabel('RViz 嵌入区域\n融合点云 / 编号关键点 / 预览留影 / 真实轨迹')
        text.setAlignment(module.QtCore.Qt.AlignCenter)
        text.setStyleSheet('color:white;font-size:21px')
        layout.addWidget(text)
    def setSplashPath(self,*args):pass
    def initialize(self):pass
    def load(self,*args):pass
    def setMenuBar(self,*args):pass
    def setStatusBar(self,*args):pass
    def setHideButtonVisibility(self,*args):pass

class Config:pass
class Reader:
    def readFile(self,*args):pass

module.rviz.VisualizationFrame=PreviewFrame
module.rviz.Config=Config
module.rviz.YamlConfigReader=Reader
module.rospy.get_param=lambda name:'fixture.rviz'
app=QtWidgets.QApplication(sys.argv)
panel=module.Panel.__new__(module.Panel)
QtWidgets.QMainWindow.__init__(panel)
panel.status={};panel.busy=False;panel.snapshot=False
panel.calibration_frozen=False;panel.pixels={};panel.images={};panel.buttons={};panel._built=False
panel.build()
panel._built=True
panel.setFixedSize(1700,950)
panel.show()
recording=root/'runtime/evidence/grasp_path_regenerated_20260922T211324'
for camera in ('rs1','rs3'):
    image=cv2.imread(str(recording/(camera+'_rgb.png')))
    if image is None:raise RuntimeError('recorded_camera_image_missing:'+camera)
    panel.apply_image(camera,cv2.cvtColor(image,cv2.COLOR_BGR2RGB),time.time())
panel.apply_status(dict(session_id='fixture',stage=1,state='PREVIEW_READY',reason='检查通过，等待执行本段',
    stage_names=['预抓取','抓取','抬升并运输','下放释放'],calibration_valid=True,
    calibration_rows=[{}]*4,feedback_age_s=.02,keypoints=[{}]*5,
    selection=[{'id':1},{'id':3}],preview=dict(id='fixture-preview',detail=dict(
        source='point_cloud_rule',goal_matrix='TCP [0.31, 0.08, 0.23]',
        goal_joints='[0.10, -0.33, 0.42, 0.0, 0.30, 0.0]',duration_s=12.5,
        audit='限位 / 自碰撞 / 环境碰撞 / 未知空间已检查',rejections=[])),
    connected=True,warning='',tracking_stamp=time.time(),tracking_valid=5,
    constraints=dict(blue_cube=dict(keypoint_id=1,position_m=[.31,.08,.12]),
                     yellow_disk=dict(keypoint_id=3,position_m=[.45,.03,.08]))))
panel.picture_state.setText('录制 RS1/RS3 图像回放；RViz 占位框；无设备测试')
panel.statusBar().showMessage('离屏回放测试：无 ROS 主站、CAN 或相机连接')
app.processEvents()
output=root/'runtime/evidence/supervised_workbench_fixture.png'
output.parent.mkdir(parents=True,exist_ok=True)
if not panel.grab().save(str(output)):
    raise RuntimeError('screenshot_save_failed')
print(output)
