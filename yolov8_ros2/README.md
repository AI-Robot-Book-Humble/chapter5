# YOLOv8とROS2のサンプルプログラム

## 概要

- YOLOv8とROS2による深層学習の物体検出プログラム
- Ubuntu 22.04, ROS Humbleで作成・確認

## インストール

- [opencv_ros2](../opencv_ros2/README.md)のインストール作業

- YOLOv8ソフトウェアをインストール
  ```
  pip3 install ultralytics
  pip3 uninstall -y opencv-python
  ```
  注：open-python は ultralytics とともに自動的にインストールされます．したがって，open-contrib-python との競合を避けるためにこれを削除する必要があります．

## 実行

5.7.2節：YOLOの物体検出
- 端末1：USBカメラのusb_camパッケージを起動
  ```
  ros2 run usb_cam usb_cam_node_exe
  ```
- 端末2：プログラムを実行
  ```
  ros2 run yolov8_ros2 object_detection
  ```
- プログラムが正常に実行されると，新たなウィンドウが現れ，検出物体に色付きの枠が描きたされたカメラ画像が表示されます．

5.7.2節：検出物体の位置推定
- 端末1：深度カメラIntel RealSense D415用のROSノードを起動
  ```
  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
  ```
- 端末2：プログラムを実行
  ```
  ros2 run yolov8_ros2 object_detection_tf
  ```
- プログラムが正常に実行されると，新たなウィンドウが現れ，深度画像に対象物体のバウンディングボックスが表示
- /tfトピックにはカメラ座標系における物体の3次元位置が出力

5.7.2節：物体検出のアクションサーバ
- 端末1：深度カメラIntel RealSense D415用のROSノードを起動
  ```
  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
  ```
- 端末2：プログラムを実行
  ```
  ros2 run yolov8_ros2 object_detection_action_server
  ```
- 端末3：ROSアクション通信を呼び出し（対象物体'cup'）
  - 対象物体を探す
    ```
    ros2 action send_goal /vision/command airobot_interfaces/action/StringCommand "{command: find cup}"
    ```
  - 対象物体を連続的に追跡
    ```
    ros2 action send_goal /vision/command airobot_interfaces/action/StringCommand "{command: track cup}"
    ```
  - 物体検出の処理を停止
    ```
    ros2 action send_goal /vision/command airobot_interfaces/action/StringCommand "{command: stop}"
    ```

## ヘルプ

- このサンプルプログラムは，Ubuntu上でしか動作が確認できていません．Windowsで開発されている方は，VirtualBox、VMwareなどのバーチャルマシンにUbuntuをインストールしてサンプルプログラムを実行する事ができます．

## 著者

タン ジェフリー トゥ チュアン　TAN Jeffrey Too Chuan

## 履歴

- 2024-10-10: Ubuntu 22.04, ROS Humbleにの更新
- 2022-08-27: ライセンス・ドキュメントの整備

## ライセンス

Copyright (c) 2022-2025, TAN Jeffrey Too Chuan  
All rights reserved.  
This project is licensed under the Apache License 2.0 found in the LICENSE file in the root directory of this project.

## 参考文献

- https://github.com/ultralytics/ultralytics
