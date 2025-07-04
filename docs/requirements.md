# Yêu cầu hệ thống - Facetory

## 📋 Yêu cầu chức năng

### 1. Upload & Phát hiện khuôn mặt
- **FR-001**: Hệ thống cho phép user upload ảnh tĩnh (JPG, PNG, max 10MB)
- **FR-002**: Tự động phát hiện số lượng khuôn mặt trong ảnh
- **FR-003**: Nếu phát hiện >1 khuôn mặt, cho phép user crop chọn 1 khuôn mặt
- **FR-004**: Hiển thị preview ảnh đã crop

### 2. Trích xuất lớp trang điểm
- **FR-005**: Nhận diện các vùng khuôn mặt (mắt, môi, má, lông mày, mũi)
- **FR-006**: Trích xuất màu sắc trang điểm từng vùng
- **FR-007**: Trích xuất kiểu dáng trang điểm (hình dạng môi, kiểu mắt...)
- **FR-008**: Hiển thị kết quả trích xuất với overlay trên ảnh

### 3. Tạo filter
- **FR-009**: Tạo filter từ thông tin trang điểm đã trích xuất
- **FR-010**: Hiển thị preview filter trên ảnh gốc
- **FR-011**: Cho phép đặt tên filter

### 4. Áp dụng filter real-time
- **FR-012**: Truy cập camera của user
- **FR-013**: Áp dụng filter lên video stream real-time
- **FR-014**: Chụp ảnh với filter đã áp dụng
- **FR-015**: Download ảnh đã chụp

### 5. Chỉnh sửa filter
- **FR-016**: Chỉnh sửa màu sắc từng vùng (môi, mắt, má...)
- **FR-017**: Điều chỉnh độ đậm nhạt của filter
- **FR-018**: Preview thay đổi real-time
- **FR-019**: Lưu filter đã chỉnh sửa

### 6. Quản lý user
- **FR-020**: Đăng ký tài khoản mới
- **FR-021**: Đăng nhập/đăng xuất
- **FR-022**: Lưu trữ ảnh gốc, filter, ảnh chụp theo user
- **FR-023**: Xem lịch sử filter và ảnh đã tạo
- **FR-024**: Xóa filter/ảnh không muốn giữ

## 🔒 Yêu cầu phi chức năng

### Hiệu năng
- **NFR-001**: Thời gian phát hiện khuôn mặt < 3 giây
- **NFR-002**: Thời gian trích xuất makeup < 5 giây
- **NFR-003**: Filter real-time chạy mượt (30fps)
- **NFR-004**: Hỗ trợ upload ảnh tối đa 10MB

### Bảo mật
- **NFR-005**: Mã hóa mật khẩu user
- **NFR-006**: JWT token cho authentication
- **NFR-007**: Rate limiting cho API calls
- **NFR-008**: Validation file upload (type, size)

### Khả năng mở rộng
- **NFR-009**: Hỗ trợ nhiều user đồng thời
- **NFR-010**: Có thể scale AI services độc lập
- **NFR-011**: Backup dữ liệu tự động

### Giao diện
- **NFR-012**: Responsive design (mobile, tablet, desktop)
- **NFR-013**: UI/UX thân thiện, dễ sử dụng
- **NFR-014**: Loading states cho các thao tác AI

## 🎯 User Stories

### Guest User (không đăng nhập)
- **US-001**: Là một guest user, tôi muốn upload ảnh để tạo filter
- **US-002**: Là một guest user, tôi muốn dùng thử filter trên camera
- **US-003**: Là một guest user, tôi muốn download ảnh đã chụp với filter

### Registered User
- **US-004**: Là một registered user, tôi muốn đăng ký tài khoản
- **US-005**: Là một registered user, tôi muốn đăng nhập vào hệ thống
- **US-006**: Là một registered user, tôi muốn lưu filter đã tạo
- **US-007**: Là một registered user, tôi muốn xem lịch sử filter của mình
- **US-008**: Là một registered user, tôi muốn chỉnh sửa filter đã lưu
- **US-009**: Là một registered user, tôi muốn xóa filter không cần thiết

## 📊 Acceptance Criteria

### Upload & Face Detection
- ✅ Upload ảnh thành công
- ✅ Hiển thị số lượng khuôn mặt phát hiện được
- ✅ Crop tool hoạt động chính xác
- ✅ Preview ảnh đã crop

### Makeup Extraction
- ✅ Phân vùng khuôn mặt chính xác
- ✅ Trích xuất màu sắc đúng
- ✅ Hiển thị overlay kết quả
- ✅ Thời gian xử lý < 5 giây

### Filter Generation
- ✅ Tạo filter từ makeup data
- ✅ Preview filter trên ảnh gốc
- ✅ Filter có thể áp dụng real-time

### Real-time Application
- ✅ Camera hoạt động trên các browser
- ✅ Filter áp dụng mượt mà
- ✅ Chụp ảnh thành công
- ✅ Download ảnh hoạt động

### User Management
- ✅ Đăng ký/đăng nhập thành công
- ✅ Lưu trữ dữ liệu theo user
- ✅ Xem lịch sử filter/ảnh
- ✅ Chỉnh sửa/xóa dữ liệu 