import cv2
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
all_pixels = 0
# 移動ノード
def __init__(self):
        super().__init__('move_robot_node')
def publish_velocity(self):
    msg = Twist()
    msg.linear.x = 0.2    # 前進（m/s）
    msg.angular.z = 0.0   # 回転なし（rad/s）
    self.publisher_.publish(msg)
def publish_stop(self):
    msg = Twist()
    msg.linear.x = 0.0    # 停止（m/s）
    msg.angular.z = 0.0   # 回転なし（rad/s）
    self.publisher_.publish(msg)
#しきい値を決定
lower_red1 = np.array([0, 0, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 0, 80])
upper_red2 = np.array([179, 255, 255])
# 移動命令
while True:
    if all_pixels < 180000:
        # 画像の読み込み
        image = cv2.imread("C:/Users/darkc/Documents/Gazo/sample_red1.png") #画像をRGBで読み込む
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #RGB→HSVに変換
         #二値化
        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
         #先ほどの二つを合わせる
        mask = cv2.bitwise_or(mask1, mask2)
         #ノイズ除去
        kernel = np.ones((10, 10), np.uint8)
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 開処理
        mask_cleaned = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel) # 閉処理  
         #ピクセル数をカウントする
        white_pixels = np.sum(mask_cleaned == 255) #白のピクセル数
        black_pixels = np.sum(mask_cleaned == 0) #黒のピクセル数
        all_pixels = white_pixels + black_pixels
        print(f"白(赤)ピクセル数: {white_pixels}")
        print(f"黒(背景)ピクセル数: {black_pixels}")
        print(f"合計ピクセル数: {all_pixels}")
        # 重心
        # 輪郭抽出
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask_cleaned.shape # 画像サイズを取得
        center_x = w // 2  # 画像の中央 x 座標

        # 赤コーンの重心（複数あれば一番大きいのを使う）
        largest_area = 0
        center_cx = None

        for cnt in contours:
            area = cv2.contourArea(cnt) # 輪郭のピクセル数の計算
            if area > largest_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  # 重心 x 座標
                    cy = int(M["m01"] / M["m00"])  # 重心 y 座標
                    center_cx = cx
                    center_cy = cy
                    largest_area = area

        # オリジナル画像に描画
        output = image.copy()

        if center_cx is not None:
            # 中央線
            cv2.line(output, (center_x, 0), (center_x, h), (255, 0, 0), 2)  # 青い線（画像の中央）
            # 重心位置
            cv2.circle(output, (center_cx, center_cy), 8, (0, 0, 255), -1)  # 赤丸（重心）
            # 距離を表示
            offset = center_cx - center_x  # 中心からのずれ（+なら右、-なら左）
            text = f"Offset: {offset}px"
            cv2.putText(output, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

             # 線で接続（オプション）
            cv2.line(output, (center_x, center_cy), (center_cx, center_cy), (0, 255, 0), 2)
        else:
            print("赤コーンが見つかりませんでした。")

        # 表示
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20,6))
        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("Center Offset Visualization")

        plt.subplot(1,3,1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.subplot(1,3,2)
        plt.title("Red Mask")
        plt.imshow(mask_cleaned, cmap='gray')

        plt.show()
        # 前進
        publish_velocity()
    else:
        # 停止
        publish_stop()
        break