class Customer:
    def __init__(self, customer_id=None, age=None, annual_income=None, spending_score=None, cluster=None):
        self.CustomerId = customer_id
        self.Age = age
        self.AnnualIncome = annual_income
        self.SpendingScore = spending_score
        self.Cluster = cluster

    def __str__(self):
        return f"Customer(ID={self.CustomerId}, Age={self.Age}, Income={self.AnnualIncome}, Score={self.SpendingScore}, Cluster={self.Cluster})"
