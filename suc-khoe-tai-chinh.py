from vnstock import Vnstock
stock = Vnstock().stock(symbol='VCI', source='VCI')
# # Bảng cân đối kế toán - năm
# print(stock.finance.balance_sheet(period='year', lang='vi', dropna=True))
# # Bảng cân đối kế toán - quý
# print(stock.finance.balance_sheet(period='quarter', lang='en', dropna=True))
# # Kết quả hoạt động kinh doanh
# print(stock.finance.income_statement(period='year', lang='vi', dropna=True))
# # Lưu chuyển tiền tệ
# print(stock.finance.cash_flow(period='year', dropna=True))
# # Chỉ số tài chính
# print(stock.finance.ratio(period='year', lang='vi', dropna=True))

# Bảng cân đối kế toán - năm
df_bs_year = stock.finance.balance_sheet(period='year', lang='vi', dropna=True)
print("\n[Bảng cân đối kế toán - năm] Columns and types:")
print(df_bs_year.dtypes)

# Bảng cân đối kế toán - quý
df_bs_quarter = stock.finance.balance_sheet(period='quarter', lang='en', dropna=True)
print("\n[Bảng cân đối kế toán - quý] Columns and types:")
print(df_bs_quarter.dtypes)

# Kết quả hoạt động kinh doanh
df_income = stock.finance.income_statement(period='year', lang='vi', dropna=True)
print("\n[Kết quả hoạt động kinh doanh] Columns and types:")
print(df_income.dtypes)

# Lưu chuyển tiền tệ
df_cashflow = stock.finance.cash_flow(period='year', dropna=True)
print("\n[Lưu chuyển tiền tệ] Columns and types:")
print(df_cashflow.dtypes)

# Chỉ số tài chính
df_ratio = stock.finance.ratio(period='year', lang='vi', dropna=True)
print("\n[Chỉ số tài chính] Columns and types:")
print(df_ratio.dtypes)


[Bảng cân đối kế toán - năm] Columns and types:
CP                                           object
Năm                                           int64
TÀI SẢN NGẮN HẠN (đồng)                       int64
Tiền và tương đương tiền (đồng)               int64
Giá trị thuần đầu tư ngắn hạn (đồng)          int64
Các khoản phải thu ngắn hạn (đồng)            int64
Hàng tồn kho ròng                             int64
Tài sản lưu động khác                         int64
TÀI SẢN DÀI HẠN (đồng)                        int64
Tài sản cố định (đồng)                        int64
Đầu tư dài hạn (đồng)                         int64
Tài sản dài hạn khác                          int64
TỔNG CỘNG TÀI SẢN (đồng)                      int64
NỢ PHẢI TRẢ (đồng)                            int64
Nợ ngắn hạn (đồng)                            int64
Nợ dài hạn (đồng)                             int64
VỐN CHỦ SỞ HỮU (đồng)                         int64
Vốn và các quỹ (đồng)                         int64
Các quỹ khác                                  int64
Lãi chưa phân phối (đồng)                     int64
TỔNG CỘNG NGUỒN VỐN (đồng)                    int64
Trả trước cho người bán ngắn hạn (đồng)       int64
Hàng tồn kho, ròng (đồng)                     int64
Tài sản lưu động khác (đồng)                  int64
Cổ phiếu phổ thông (đồng)                   float64
Vốn góp của chủ sở hữu (đồng)                 int64
Vay và nợ thuê tài chính dài hạn (đồng)       int64
Người mua trả tiền trước ngắn hạn (đồng)      int64
Vay và nợ thuê tài chính ngắn hạn (đồng)      int64
Trả trước dài hạn (đồng)                      int64
Tài sản dài hạn khác (đồng)                   int64
LỢI ÍCH CỦA CỔ ĐÔNG THIỂU SỐ                float64
dtype: object

[Bảng cân đối kế toán - quý] Columns and types:
ticker                                    object
yearReport                                 int64
lengthReport                               int64
CURRENT ASSETS (Bn. VND)                   int64
Cash and cash equivalents (Bn. VND)        int64
Short-term investments (Bn. VND)           int64
Accounts receivable (Bn. VND)              int64
Net Inventories                            int64
Other current assets                       int64
LONG-TERM ASSETS (Bn. VND)                 int64
Long-term trade receivables (Bn. VND)      int64
Fixed assets (Bn. VND)                     int64
Long-term investments (Bn. VND)            int64
Other non-current assets                   int64
TOTAL ASSETS (Bn. VND)                     int64
LIABILITIES (Bn. VND)                      int64
Current liabilities (Bn. VND)              int64
Long-term liabilities (Bn. VND)            int64
OWNER'S EQUITY(Bn.VND)                     int64
Capital and reserves (Bn. VND)             int64
Other Reserves                             int64
Undistributed earnings (Bn. VND)           int64
Minority Interest                          int64
TOTAL RESOURCES (Bn. VND)                  int64
Prepayments to suppliers (Bn. VND)         int64
Inventories, Net (Bn. VND)                 int64
Other current assets (Bn. VND)             int64
Common shares (Bn. VND)                  float64
Paid-in capital (Bn. VND)                  int64
Long-term borrowings (Bn. VND)             int64
Advances from customers (Bn. VND)          int64
Short-term borrowings (Bn. VND)            int64
Long-term prepayments (Bn. VND)            int64
Other long-term assets (Bn. VND)           int64
Goodwill                                   int64
MINORITY INTERESTS                       float64
dtype: object

[Kết quả hoạt động kinh doanh] Columns and types:
CP                                                   object
Năm                                                   int64
Doanh thu (đồng)                                      int64
Tăng trưởng doanh thu (%)                           float64
Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)      int64
Tăng trưởng lợi nhuận (%)                           float64
Doanh thu thuần                                       int64
Giá vốn hàng bán                                      int64
Lãi gộp                                               int64
Thu nhập tài chính                                  float64
Chi phí tài chính                                   float64
Chi phí tiền lãi vay                                float64
Chi phí quản lý DN                                    int64
Lãi/Lỗ từ hoạt động kinh doanh                        int64
Lợi nhuận khác                                        int64
LN trước thuế                                         int64
Chi phí thuế TNDN hiện hành                           int64
Chi phí thuế TNDN hoãn lại                            int64
Lợi nhuận thuần                                       int64
Cổ đông thiểu số                                      int64
Cổ đông của Công ty mẹ                                int64
Doanh thu bán hàng và cung cấp dịch vụ                int64
Thu nhập khác                                         int64
Thu nhập/Chi phí khác                                 int64
dtype: object

[Lưu chuyển tiền tệ] Columns and types:
ticker                                                                      object
yearReport                                                                   int64
Net Profit/Loss before tax                                                   int64
Depreciation and Amortisation                                                int64
Provision for credit losses                                                  int64
Unrealized foreign exchange gain/loss                                        int64
Profit/Loss from investing activities                                        int64
Interest Expense                                                             int64
Interest income and dividends                                              float64
Operating profit before changes in working capital                           int64
Increase/Decrease in receivables                                           float64
Increase/Decrease in inventories                                             int64
Increase/Decrease in payables                                              float64
Increase/Decrease in prepaid expenses                                        int64
Interest paid                                                                int64
Business Income Tax paid                                                     int64
Other receipts from operating activities                                     int64
Other payments on operating activities                                       int64
Net cash inflows/outflows from operating activities                          int64
Purchase of fixed assets                                                     int64
Proceeds from disposal of fixed assets                                       int64
Loans granted, purchases of debt instruments (Bn. VND)                       int64
Collection of loans, proceeds from sales of debts instruments (Bn. VND)      int64
Investment in other entities                                                 int64
Proceeds from divestment in other entities                                   int64
Gain on Dividend                                                             int64
Net Cash Flows from Investing Activities                                     int64
Increase in charter captial                                                  int64
Payments for share repurchases                                               int64
Proceeds from borrowings                                                     int64
Repayment of borrowings                                                      int64
Dividends paid                                                               int64
Cash flows from financial activities                                         int64
Net increase/decrease in cash and cash equivalents                           int64
Cash and cash equivalents                                                    int64
Cash and Cash Equivalents at the end of period                               int64
_Increase/Decrease in receivables                                            int64
_Increase/Decrease in payables                                               int64
dtype: object

[Chỉ số tài chính] Columns and types:
Meta                         CP                               object
                             Năm                               int64
                             Kỳ                                int64
Chỉ tiêu cơ cấu nguồn vốn    (Vay NH+DH)/VCSH                float64
                             Nợ/VCSH                         float64
                             TSCĐ / Vốn CSH                  float64
                             Vốn CSH/Vốn điều lệ             float64
Chỉ tiêu hiệu quả hoạt động  Vòng quay tài sản               float64
                             Vòng quay TSCĐ                  float64
                             Số ngày thu tiền bình quân      float64
                             Số ngày tồn kho bình quân       float64
                             Số ngày thanh toán bình quân    float64
                             Chu kỳ tiền                     float64
                             Vòng quay hàng tồn kho          float64
Chỉ tiêu khả năng sinh lợi   Biên EBIT (%)                   float64
                             Biên lợi nhuận gộp (%)          float64
                             Biên lợi nhuận ròng (%)         float64
                             ROE (%)                         float64
                             ROIC (%)                        float64
                             ROA (%)                         float64
                             EBITDA (Tỷ đồng)                  int64
                             EBIT (Tỷ đồng)                  float64
                             Tỷ suất cổ tức (%)              float64
Chỉ tiêu thanh khoản         Chỉ số thanh toán hiện thời     float64
                             Chỉ số thanh toán tiền mặt      float64
                             Chỉ số thanh toán nhanh         float64
                             Khả năng chi trả lãi vay        float64
                             Đòn bẩy tài chính               float64
Chỉ tiêu định giá            P/B                             float64
                             Vốn hóa (Tỷ đồng)                 int64
                             Số CP lưu hành (Triệu CP)         int64
                             P/E                             float64
                             P/S                             float64
                             P/Cash Flow                     float64
                             EPS (VND)                       float64
                             BVPS (VND)                      float64
                             EV/EBITDA                       float64