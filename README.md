# Dự án: Group04_Data_Analysis_Project
1. Giới thiệu dự án:
1.1.Bối cảnh và đặt vấn đề:
   Đồng bằng sông Cửu Long (ĐBSCL) là "vựa lúa" lớn nhất Việt Nam, đóng góp hơn 50% sản lượng lúa gạo cả nước. Tuy nhiên, nông dân tại đây thường xuyên đối mặt với tình trạng "được mùa mất giá" và sự nhiễu loạn thông tin thị trường. Dữ liệu về giá lúa tươi (tại ruộng), giá gạo bán lẻ, chỉ số chi phí (phân bón) và biến động thời tiết hiện đang nằm rải rác ở nhiều nguồn khác nhau, gây khó khăn cho việc theo dõi và đưa ra quyết định bán hàng tối ưu.

# Xây dựng Data Warehouse Tích Hợp Dự Báo Giá Gạo Nội Địa - ĐBSCL

## 1. Giới thiệu dự án (Project Overview)

### 1.1. Bối cảnh và Đặt vấn đề
Đồng bằng sông Cửu Long (ĐBSCL) là "vựa lúa" lớn nhất Việt Nam, đóng góp hơn 50% sản lượng lúa gạo cả nước. Tuy nhiên, nông dân tại đây thường xuyên đối mặt với tình trạng "được mùa mất giá" và sự nhiễu loạn thông tin thị trường. Dữ liệu về giá lúa tươi (tại ruộng), giá gạo bán lẻ, chỉ số chi phí (phân bón) và biến động thời tiết hiện đang nằm rải rác ở nhiều nguồn khác nhau, gây khó khăn cho việc theo dõi và đưa ra quyết định bán hàng tối ưu.

### 1.2. Mục tiêu dự án
* **Xây dựng Kho dữ liệu (Data Warehouse):** Tích hợp dữ liệu từ các nguồn mở (VnEconomy, Thuonghieucongluan, Yimex, Open-Meteo).
* **Phân tích đa chiều:** Đánh giá tác động của lạm phát (CPI), chi phí đầu vào và thời tiết.
* **Dự báo thông minh:** Sử dụng thuật toán ARIMA_PLUS trên BigQuery ML.

### 1.3. Phạm vi nghiên cứu
* **Đối tượng:** Giá lúa tươi (OM 5451, Đài Thơm 8, IR50404) 
* **Khu vực:** khu vực Đồng bằng sông Cửu Long.
* **Thời gian:** Từ năm 2024 đến tháng nay

## 2. Hướng dẫn cài đặt và chạy code

### 2.1. Yêu cầu hệ thống
* Ngôn ngữ: Python 3.9 trở lên.
* Tài khoản Google Cloud Platform (GCP) đã cấu hình BigQuery.

### 2.2. Các bước cài đặt
Cài đặt các thư viện cần thiết bằng lệnh:
`pip install -r requirements.txt`

### 2.3. Hướng dẫn chạy code
1. **Bước 1 (Thu thập):** Chạy `Group04_Data_Analysis_Project/notebooks/01_crawling_data.ipynb` để crawl dữ liệu cũ. Để duy trì và cập nhật dữ liệu tự động thì thực thi các file zip `Group04_Data_Analysis_Project/src/data_processing.py` trong Cloud Run Functions.
2. **Bước 2 (Xử lý): Thực thi các file sql trong `Group04_Data_Analysis_Project/src/clean.sql` trên BigQuery để chuẩn hóa, xử lý dữ liệu.
3. **Bước 3 (Huấn luyện):** Chạy `Group04_Data_Analysis_Project/src/train.py` để luyện mô hình.
4. **Bước 4 (Dự báo):** Chạy `Group04_Data_Analysis_Project/src/predict.py` để lấy kết quả.
5. **Bước 5 (Trực quan hóa dữ liệu):** Chạy `Group04_Data_Analysis_Project/reports/figures` để xem kết quả trực quan hóa.

