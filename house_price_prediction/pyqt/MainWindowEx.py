from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QTableWidgetItem
from MainWindow import Ui_MainWindow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import sys
import os

# Add parent directory to path to import FileUtil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FileUtil import FileUtil


class MainWindowEx(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialize variables
        self.lm = None
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Set default values
        self.ui.lineEdit_Select.setText("../USA_Housing.csv")
        self.ui.lineEdit_TrainingRate.setText("80")

        # Connect signals to slots
        self.ui.pushButton_PickData.clicked.connect(self.pickDataset)
        self.ui.pushButton_TrainModel.clicked.connect(self.trainModel)
        self.ui.pushButton_EvaluateModel.clicked.connect(self.evaluateModel)
        self.ui.pushButton_SaveModel.clicked.connect(self.saveModel)
        self.ui.pushButton_LoadModel.clicked.connect(self.loadModel)
        self.ui.pushButton_Predict.clicked.connect(self.predict)

    def pickDataset(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filename:
            self.ui.lineEdit_Select.setText(filename)

    def trainModel(self):
        try:
            # Get dataset path
            dataset_path = self.ui.lineEdit_Select.text()

            # Load dataset
            self.df = pd.read_csv(dataset_path)

            # Prepare features and target
            self.X = self.df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                              'Avg. Area Number of Bedrooms', 'Area Population']]
            self.y = self.df['Price']

            # Get training rate
            training_rate = float(self.ui.lineEdit_TrainingRate.text()) / 100

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=1-training_rate, random_state=101
            )

            # Train model
            self.lm = LinearRegression()
            self.lm.fit(self.X_train, self.y_train)

            # Show data in table
            self.displayData()

            QMessageBox.information(self, "Success", "Model trained successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")

    def displayData(self):
        # Display first 20 rows in table
        self.ui.tableWidget_Data.setRowCount(min(20, len(self.df)))

        for i in range(min(20, len(self.df))):
            self.ui.tableWidget_Data.setItem(i, 0, QTableWidgetItem(str(self.df.iloc[i, 0])))
            self.ui.tableWidget_Data.setItem(i, 1, QTableWidgetItem(str(self.df.iloc[i, 1])))
            self.ui.tableWidget_Data.setItem(i, 2, QTableWidgetItem(str(self.df.iloc[i, 2])))
            self.ui.tableWidget_Data.setItem(i, 3, QTableWidgetItem(str(self.df.iloc[i, 3])))
            self.ui.tableWidget_Data.setItem(i, 4, QTableWidgetItem(str(self.df.iloc[i, 4])))
            self.ui.tableWidget_Data.setItem(i, 5, QTableWidgetItem(str(self.df.iloc[i, 5])))

    def evaluateModel(self):
        try:
            if self.lm is None:
                QMessageBox.warning(self, "Warning", "Please train the model first!")
                return

            # Make predictions
            predictions = self.lm.predict(self.X_test)

            # Display coefficients
            self.ui.tableWidget_Coeff.setRowCount(6)

            # Intercept
            self.ui.tableWidget_Coeff.setItem(0, 0, QTableWidgetItem("Intercept"))
            self.ui.tableWidget_Coeff.setItem(0, 1, QTableWidgetItem(f"{self.lm.intercept_:.2f}"))

            # Coefficients
            feature_names = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                           'Avg. Area Number of Bedrooms', 'Area Population']

            for i, (name, coef) in enumerate(zip(feature_names, self.lm.coef_), 1):
                self.ui.tableWidget_Coeff.setItem(i, 0, QTableWidgetItem(name))
                self.ui.tableWidget_Coeff.setItem(i, 1, QTableWidgetItem(f"{coef:.2f}"))

            # Calculate metrics
            mae = metrics.mean_absolute_error(self.y_test, predictions)
            mse = metrics.mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)

            # Display metrics
            self.ui.lineEdit_MAE.setText(f"{mae:.2f}")
            self.ui.lineEdit_MSE.setText(f"{mse:.2f}")
            self.ui.lineEdit_RMSE.setText(f"{rmse:.2f}")

            # Add predictions to table
            for i in range(min(20, len(self.X_test))):
                if i < self.ui.tableWidget_Data.rowCount():
                    self.ui.tableWidget_Data.setItem(i, 6, QTableWidgetItem(f"{predictions[i]:.2f}"))

            QMessageBox.information(self, "Success", "Model evaluated successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Evaluation failed: {str(e)}")

    def saveModel(self):
        try:
            if self.lm is None:
                QMessageBox.warning(self, "Warning", "Please train the model first!")
                return

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model",
                "housingmodel.zip",
                "Zip Files (*.zip);;All Files (*.*)"
            )

            if filename:
                FileUtil.savemodel(self.lm, filename)
                QMessageBox.information(self, "Success", f"Model saved to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

    def loadModel(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load Model",
                "",
                "Zip Files (*.zip);;All Files (*.*)"
            )

            if filename:
                self.lm = FileUtil.loadmodel(filename)
                if self.lm is not None:
                    QMessageBox.information(self, "Success", f"Model loaded from {filename}")
                else:
                    QMessageBox.critical(self, "Error", "Failed to load model")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {str(e)}")

    def predict(self):
        try:
            if self.lm is None:
                QMessageBox.warning(self, "Warning", "Please train or load a model first!")
                return

            # Get input values
            area_income = float(self.ui.lineEdit_Income.text())
            area_house_age = float(self.ui.lineEdit_HouseAge.text())
            area_rooms = float(self.ui.lineEdit_Rooms.text())
            area_bedrooms = float(self.ui.lineEdit_Bedrooms.text())
            area_population = float(self.ui.lineEdit_Population.text())

            # Make prediction
            result = self.lm.predict([[area_income, area_house_age, area_rooms,
                                     area_bedrooms, area_population]])

            # Display result
            self.ui.lineEdit_Pred.setText(f"{result[0]:.2f}")

        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numbers!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
