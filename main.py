from ultralytics import YOLO
import cv2
import requests
import datetime
import threading


# Load model tốt nhất đã được train
model = YOLO(r'model.pt')

# Khởi tạo camera, 0 mặc định là camera ở local
cam = cv2.VideoCapture(r'test.mp4')

# ===== CẤU HÌNH TELEGRAM BOT =====
# THAY ĐỔI API_KEY VÀ CHAT_ID CỦA BẠN Ở ĐÂY
api_key = '8065117282:AAEOWhpBmVSLtHCsHZWqfef4Q6z0le94Ctc'  # Thay bằng token từ @BotFather
chat_id = '7874082485'    # Thay bằng chat ID của bạn

# Biến theo dõi để tránh spam (chỉ gửi 1 lần mỗi 10 giây)
last_alert_time = 0

def send_telegram_message():
    """Gửi cảnh báo qua Telegram"""
    global last_alert_time
    current_time = datetime.datetime.now()

    # Kiểm tra đã gửi trong vòng 10 giây chưa
    if (current_time.timestamp() - last_alert_time) < 10:
        return

    # Tạo message với timestamp
    message = f'🔥 CẢNH BÁO: Phát hiện cháy lúc {current_time.strftime("%H:%M:%S - %d/%m/%Y")}'
    url = f'https://api.telegram.org/bot{api_key}/sendMessage'

    try:
        response = requests.get(url, params={
            'chat_id': chat_id,
            'text': message
        })
        if response.status_code == 200:
            print(f"✓ Đã gửi cảnh báo Telegram: {current_time.strftime('%H:%M:%S')}")
            last_alert_time = current_time.timestamp()
        else:
            print(f"✗ Lỗi gửi Telegram: {response.status_code}")
    except Exception as e:
        print(f"✗ Lỗi kết nối Telegram: {e}")

while True:
    # Đọc ảnh
    check, cap = cam.read()
    # cap = cv2.flip(cap, 1)
    if not check:
        print("Không thể đọc từ camera.")
        break
    
    # Dự đoán 
    res = model.predict(source= cap, conf = 0.4, verbose= False) # verbose = False để tắt các dòng thông báo thừa

    # Lấy ra độ tin cậy của từng box
    conf = res[0].boxes.conf
    # Lấy ra data về các box
    boxes = res[0].boxes.xyxy

    # Vẽ và hiển thị thông qua cv2
    for i in range(len(boxes)):
        # Ép kiểu về số nguyên do yêu cầu thông số của cv2 qui định
        x1, y1, x2, y2 = map(int, boxes[i]) # Lấy tọa độ box
        # Vẽ box
        cv2.rectangle(cap, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Điền độ tin cậy của box
        cv2.putText(cap, f'{round(float(conf[i]), 2)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # vẽ tâm box
        x, y = (x1 + x2) //2, (y1 + y2) // 2 
        color = (0, 255, 0)             
        cv2.circle(cap, (x, y), radius=3, color=color, thickness=-1)  # thickness=-1 để tô đầy hình tròn
    
    # Thông báo về số ngọn lửa được tìm thấy
    fire = len(boxes)
    if(fire != 0):
        # Hiển thị cảnh báo màu đỏ
        cv2.putText(cap, f'{fire} fire detected', (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # GỬI CẢNH BÁO QUA TELEGRAM KHI CÓ LỬA
        if(api_key != 'YOUR_API_TOKEN_HERE'):  # Chỉ gửi nếu đã cấu hình
            # Thông báo qua luồng riêng biệt để tránh delay trong vòng lặp
            threading.Thread(target=send_telegram_message, daemon=True).start()
    else:
        # Hiển thị trạng thái an toàn màu xanh
        cv2.putText(cap, f'No fire visible in the observation area', (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Hiển thị ra màn hình
    cv2.imshow('fire', cap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Giải phóng camera
cam.release()
cv2.destroyAllWindows()