import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px

from project_retail.connectors.connector import Connector

# Kết nối database và lấy dữ liệu
conn = Connector(database="salesdatabase")
conn.connect()

sql = 'select distinct customer.CustomerId, Age, Annual_Income, Spending_Score ' \
      'from customer, customer_spend_score ' \
      'where customer.CustomerId=customer_spend_score.CustomerId'
df = conn.queryDataset(sql)
df.columns = ['CustomerId', 'Age', 'Annual Income', 'Spending Score']

def runKMeans(X, cluster):
    """Chạy KMeans clustering"""
    model = KMeans(n_clusters=cluster,
                   init='k-means++',
                   max_iter=500,
                   random_state=42)

    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.fit_predict(X)
    return y_kmeans, centroids, labels


# HÀM 1: Lọc và in kết quả customers theo cluster ra console
def filter_and_print_customers_by_cluster(df, cluster_number):
    """
    Lọc customers theo cluster, in ra console và trả về DataFrame

    Parameters:
    -----------
    df : DataFrame
        DataFrame chứa thông tin customers và cột 'Cluster'
    cluster_number : int
        Số thứ tự của cluster cần lọc (bắt đầu từ 0)

    Returns:
    --------
    DataFrame
        DataFrame chứa các customers thuộc cluster được chỉ định
    """
    if 'Cluster' not in df.columns:
        print("Error: DataFrame chưa có cột 'Cluster'. Vui lòng chạy KMeans trước!")
        return None

    filtered_df = df[df['Cluster'] == cluster_number]

    if filtered_df.empty:
        print(f"\nKhông có customer nào trong Cluster {cluster_number}")
        return None

    print(f"\n{'='*80}")
    print(f"DANH SÁCH CUSTOMERS THUỘC CLUSTER {cluster_number}")
    print(f"{'='*80}")
    print(f"Tổng số customers: {len(filtered_df)}")
    print(f"{'-'*80}")

    # Hiển thị header
    print(f"{'ID':<10} {'Age':<10} {'Annual Income':<20} {'Spending Score':<20}")
    print(f"{'-'*80}")

    # Hiển thị từng customer
    for idx, row in filtered_df.iterrows():
        print(f"{int(row['CustomerId']):<10} {int(row['Age']):<10} "
              f"{row['Annual Income']:<20} {int(row['Spending Score']):<20}")

    print(f"{'='*80}\n")

    # Thống kê
    print(f"THỐNG KÊ CLUSTER {cluster_number}:")
    print(f"  - Tuổi trung bình: {filtered_df['Age'].mean():.2f}")
    print(f"  - Thu nhập trung bình: {filtered_df['Annual Income'].mean():.2f}")
    print(f"  - Spending Score trung bình: {filtered_df['Spending Score'].mean():.2f}")
    print(f"{'='*80}\n")

    return filtered_df


# HÀM 2: Hiển thị bảng customers trên web (dùng Plotly)
def show_customers_on_web(filtered_df, cluster_number):
    """
    Hiển thị bảng customers theo cluster trên web browser (dùng Plotly)

    Parameters:
    -----------
    filtered_df : DataFrame
        DataFrame đã được lọc theo cluster (kết quả từ hàm 1)
    cluster_number : int
        Số thứ tự của cluster cần hiển thị
    """
    import plotly.graph_objects as go

    if filtered_df is None or filtered_df.empty:
        print(f"Không có customer nào trong Cluster {cluster_number}")
        return

    # Tạo dữ liệu cho bảng
    table_data = [
        filtered_df['CustomerId'].astype(int).tolist(),
        filtered_df['Age'].astype(int).tolist(),
        filtered_df['Annual Income'].tolist(),
        filtered_df['Spending Score'].astype(int).tolist(),
        filtered_df['Cluster'].astype(int).tolist()
    ]

    # Tính thống kê
    avg_age = filtered_df['Age'].mean()
    avg_income = filtered_df['Annual Income'].mean()
    avg_score = filtered_df['Spending Score'].mean()

    # Tạo header và cells với màu sắc
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Customer ID</b>', '<b>Age</b>', '<b>Annual Income</b>',
                    '<b>Spending Score</b>', '<b>Cluster</b>'],
            fill_color='#1976d2',
            font=dict(color='white', size=14),
            align='left',
            height=40
        ),
        cells=dict(
            values=table_data,
            fill_color='lavender',
            align='left',
            font=dict(size=12),
            height=30
        )
    )])

    # Cập nhật layout với thông tin thống kê
    title_text = f"<b>DANH SÁCH CUSTOMERS - CLUSTER {cluster_number}</b><br>" \
                 f"<i>Tổng số: {len(filtered_df)} customers</i><br>" \
                 f"Tuổi TB: {avg_age:.2f} | Thu nhập TB: {avg_income:.2f} | " \
                 f"Spending Score TB: {avg_score:.2f}"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=600
    )

    # Mở trình duyệt và hiển thị
    fig.show()


# ===== DEMO SỬ DỤNG =====

# Chạy KMeans với k=6 (Age x Annual Income x Spending Score)
print("\n=== K-MEANS VỚI K=6 (3 chiều) ===")
columns = ['Age', 'Annual Income', 'Spending Score']
X = df.loc[:, columns].values
y_kmeans, centroids, labels = runKMeans(X, 6)
df['Cluster'] = labels

# Lọc và hiển thị customers theo từng cluster của k=6 (console + web)
# Hiển thị web cho cluster từ 0 đến 5 (tổng 6 clusters)
for cluster_num in range(6):
    filtered = filter_and_print_customers_by_cluster(df, cluster_num)
    if filtered is not None:
        show_customers_on_web(filtered, cluster_num)

