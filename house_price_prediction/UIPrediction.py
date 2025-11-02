from tkinter import *
from tkinter import messagebox, ttk
from tkinter.font import Font
from tkinter import filedialog as fd

from DataSetViewer import DataSetViewer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

from FileUtil import FileUtil


class UIPrediction:
    fileName = ""

    def __init__(self):
        pass

    def create_ui(self):
        self.root = Tk()
        self.root.title("House Pricing Prediction- Faculty of Information Systems")
        self.root.geometry("1500x850")
        main_panel = PanedWindow(self.root)
        main_panel["bg"] = "yellow"
        main_panel.pack(fill=BOTH, expand=True)

        top_panel = PanedWindow(main_panel, height=80)
        top_panel["bg"] = "blue"
        main_panel.add(top_panel)
        top_panel.pack(fill=X, side=TOP, expand=False)

        font = Font(family="tahoma", size=18)
        title_label = Label(top_panel, text='House Pricing Prediction', font=font)
        title_label["bg"] = "yellow"
        top_panel.add(title_label)

        center_panel = PanedWindow(main_panel)
        main_panel.add(center_panel)
        center_panel["bg"] = "pink"
        center_panel.pack(fill=BOTH, expand=True)

        choose_dataset_panel = PanedWindow(center_panel, height=30)
        center_panel.add(choose_dataset_panel)
        choose_dataset_panel["bg"] = "orange"
        choose_dataset_panel.pack(fill=X)

        dataset_label = Label(choose_dataset_panel, text="Select Dataset:")
        self.selectedFileName = StringVar()
        self.selectedFileName.set("USA_Housing.csv")
        self.choose_dataset_entry = Entry(choose_dataset_panel, textvariable=self.selectedFileName)
        self.choose_dataset_button = Button(choose_dataset_panel, text="1.Pick Dataset",
                                            width=30, command=self.do_pick_data)

        self.view_dataset_button = Button(choose_dataset_panel, text="2.View Dataset", width=20,
                                          command=self.do_view_dataset)

        choose_dataset_panel.add(dataset_label)
        choose_dataset_panel.add(self.choose_dataset_entry)
        choose_dataset_panel.add(self.choose_dataset_button)
        choose_dataset_panel.add(self.view_dataset_button)
        self.view_dataset_button.pack(side=RIGHT, expand=False)
        self.choose_dataset_button.pack(side=RIGHT, expand=False)

        # Training Rate
        training_rate_panel = PanedWindow(center_panel, height=30)
        center_panel.add(training_rate_panel)
        training_rate_panel.pack(fill=X)
        training_rate_label = Label(training_rate_panel, text="Training Rate:")
        self.training_rate = IntVar()
        self.training_rate.set(80)
        self.training_rate_entry = Entry(training_rate_panel, textvariable=self.training_rate, width=20)
        training_rate_panel.add(training_rate_label)
        training_rate_panel.add(self.training_rate_entry)
        percent_label = Label(text="%", width=20, anchor="w", justify=LEFT)
        percent_label.pack(side=RIGHT, expand=False, fill=X)
        training_rate_panel.add(percent_label)
        self.train_model_button = Button(training_rate_panel, text="3.Train Model", width=20, command=self.do_train)
        training_rate_panel.add(self.train_model_button)
        self.evaluate_model_button = Button(training_rate_panel, text="4.Evaluate Model",
                                            width=20, command=self.do_evaluation)
        training_rate_panel.add(self.evaluate_model_button)
        self.status = StringVar()
        self.train_model_result_label = Label(training_rate_panel, textvariable=self.status)
        training_rate_panel.add(self.train_model_result_label)

        evaluate_panel = PanedWindow(center_panel, height=400)
        evaluate_panel["bg"] = "cyan"
        center_panel.add(evaluate_panel)
        evaluate_panel.pack(side=TOP, fill=BOTH, expand=True)

        coefficient_panel = PanedWindow(evaluate_panel, width=200)
        evaluate_panel.add(coefficient_panel)
        coefficient_panel.pack(side=LEFT, fill=BOTH, expand=True)

        coefficient_detail_panel = PanedWindow(coefficient_panel)
        coefficient_panel.add(coefficient_detail_panel)
        coefficient_detail_panel.pack(fill=BOTH, side=TOP, expand=True)

        self.coefficient_detail_text = Text(coefficient_detail_panel, height=12, width=50)
        scroll = Scrollbar(coefficient_detail_panel)
        self.coefficient_detail_text.configure(yscrollcommand=scroll.set)
        self.coefficient_detail_text.pack(side=LEFT, expand=False, fill=X)

        scroll.config(command=self.coefficient_detail_text.yview)
        scroll.pack(side=RIGHT, fill=Y, expand=True)

        metric_panel = PanedWindow(coefficient_panel, height=30)
        coefficient_panel.add(metric_panel)
        metric_panel.pack(side=TOP, fill=BOTH, expand=True)

        self.mae_value = DoubleVar()
        mae_label = Label(metric_panel, text="Mean Absolute Error(MAE):")
        mae_label.grid(row=0, column=0)
        mae_entry = Entry(metric_panel, text="", width=20, textvariable=self.mae_value)
        mae_entry.grid(row=0, column=1)

        self.mse_value = DoubleVar()
        mse_label = Label(metric_panel, text="Mean Square Error(MSE):")
        mse_label.grid(row=1, column=0)
        mse_entry = Entry(metric_panel, text="", width=20, textvariable=self.mse_value)
        mse_entry.grid(row=1, column=1)

        self.rmse_value = DoubleVar()
        rmse_label = Label(metric_panel, text="Root Mean Square Error(RMSE):")
        rmse_label.grid(row=2, column=0)
        rmse_entry = Entry(metric_panel, text="", width=20, textvariable=self.rmse_value)
        rmse_entry.grid(row=2, column=1)

        savemodel_button = Button(metric_panel, text="5. Save Model", width=20, command=self.do_save_model)
        savemodel_button.grid(row=3, column=1)

        loadmodel_panel = PanedWindow(center_panel, height=20)
        loadmodel_panel["bg"] = "yellow"
        loadmodel_panel.pack(fill=BOTH, side=TOP)

        loadmodel_label = Label(loadmodel_panel, text="6. Load Model:")
        loadmodel_label.grid(row=0, column=0)

        # Model selection dropdown
        self.model_options = ["housingmodel.zip", "model1.zip", "model2.zip", "model3.zip"]
        self.selected_model = StringVar()
        self.selected_model.set(self.model_options[0])  # default value

        model_menu = OptionMenu(loadmodel_panel, self.selected_model, *self.model_options)
        model_menu.grid(row=0, column=1)

        loadmodel_button = Button(loadmodel_panel, text="Load Model", command=self.do_load_model)
        loadmodel_button.grid(row=0, column=2)

        input_prediction_panel = PanedWindow(center_panel)
        input_prediction_panel.pack(fill=BOTH, side=TOP, expand=True)

        area_income_label = Label(input_prediction_panel, text="Avg. Area Income:")
        area_income_label.grid(row=0, column=0)
        self.area_income_value = DoubleVar()
        area_income_entry = Entry(input_prediction_panel, text="", width=40, textvariable=self.area_income_value)
        area_income_entry.grid(row=0, column=1)

        area_house_age_label = Label(input_prediction_panel, text="Avg. Area House Age:")
        area_house_age_label.grid(row=1, column=0)
        self.area_house_age_value = DoubleVar()
        area_house_age_entry = Entry(input_prediction_panel, text="", width=40, textvariable=self.area_house_age_value)
        area_house_age_entry.grid(row=1, column=1)

        area_number_of_rooms_label = Label(input_prediction_panel, text="Avg. Area Number of Rooms:")
        area_number_of_rooms_label.grid(row=2, column=0)
        self.area_number_of_rooms_value = DoubleVar()
        area_number_of_rooms_entry = Entry(input_prediction_panel, text="", width=40,
                                           textvariable=self.area_number_of_rooms_value)
        area_number_of_rooms_entry.grid(row=2, column=1)

        area_number_of_bedrooms_label = Label(input_prediction_panel, text="Avg. Area Number of Bedrooms:")
        area_number_of_bedrooms_label.grid(row=3, column=0)
        self.area_number_of_bedrooms_value = DoubleVar()
        area_number_of_bedrooms_entry = Entry(input_prediction_panel, text="", width=40,
                                              textvariable=self.area_number_of_bedrooms_value)
        area_number_of_bedrooms_entry.grid(row=3, column=1)

        area_population_label = Label(input_prediction_panel, text="Area Population:")
        area_population_label.grid(row=4, column=0)
        self.area_population_value = DoubleVar()
        area_population_entry = Entry(input_prediction_panel, text="", width=40,
                                      textvariable=self.area_population_value)
        area_population_entry.grid(row=4, column=1)

        prediction_button = Button(input_prediction_panel, text="7. Prediction House Pricing", command=self.do_prediction)
        prediction_button.grid(row=5, column=1)

        prediction_price_label = Label(input_prediction_panel, text="Prediction Price:")
        prediction_price_label.grid(row=6, column=0)
        self.prediction_price_value = DoubleVar()
        prediction_price_entry = Entry(input_prediction_panel, text="", width=40,
                                       textvariable=self.prediction_price_value)
        prediction_price_entry.grid(row=6, column=1)

        designedby_panel = PanedWindow(main_panel, height=20)
        designedby_panel["bg"] = "cyan"
        designedby_panel.pack(fill=BOTH, side=BOTTOM)
        designedby_label = Label(designedby_panel, text="Designed by: Hồ Quang Vinh")
        designedby_label["bg"] = "cyan"
        designedby_label.pack(side=LEFT)

        pass

    def show_ui(self):
        self.root.mainloop()

    def do_pick_data(self):
        filetypes = (("Dataset CSV", "*.csv"),
                     ("All Files", "*.*"))

        s = fd.askopenfilename(
            title="Choose dataset",
            initialdir="/",
            filetypes=filetypes)
        self.selectedFileName.set(s)
        pass

    def do_view_dataset(self):
        viewer = DataSetViewer()
        viewer.create_ui()
        viewer.show_data_listview(self.selectedFileName.get())
        viewer.show_ui()
        pass

    def do_train(self):
        from sklearn.linear_model import LinearRegression

        ratio = self.training_rate.get() / 100
        self.df = pd.read_csv(self.selectedFileName.get())

        self.X = self.df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                          'Avg. Area Number of Bedrooms', 'Area Population']]
        self.y = self.df['Price']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=1-ratio, random_state=101)

        self.lm = LinearRegression()

        self.lm.fit(self.X_train, self.y_train)
        self.status.set("Trained is finished")
        messagebox.showinfo("info", "Trained is finished")
        pass

    def do_evaluation(self):
        # print the intercept
        print(self.lm.intercept_)
        insert_text = self.lm.intercept_

        self.coeff_df = pd.DataFrame(self.lm.coef_, self.X.columns, columns=['Coefficient'])
        print(self.coeff_df)
        self.coefficient_detail_text.insert(END, self.coeff_df)

        predictions = self.lm.predict(self.X_test)
        print(predictions)
        print("self.X_test")
        print(self.X_test)
        print("self.y_test:")
        print(self.y_test)

        print('MAE:', metrics.mean_absolute_error(self.y_test, predictions))
        print('MSE:', metrics.mean_squared_error(self.y_test, predictions))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, predictions)))

        self.mae_value.set(metrics.mean_absolute_error(self.y_test, predictions))
        self.mse_value.set(metrics.mean_squared_error(self.y_test, predictions))
        self.rmse_value.set(np.sqrt(metrics.mean_squared_error(self.y_test, predictions)))

        self.status.set("Evaluation is finished")
        messagebox.showinfo("info", "Evaluation is finished")
        pass

    def do_save_model(self):
        FileUtil.savemodel(self.lm, "housingmodel.zip")
        messagebox.showinfo("info", "exported model to disk successful!")
        pass

    def do_load_model(self):
        import os
        model_filename = self.selected_model.get()

        # Check if file exists
        if not os.path.exists(model_filename):
            messagebox.showerror("error", f"Model file '{model_filename}' not found!\nPlease train and save a model first.")
            return

        self.lm = FileUtil.loadmodel(model_filename)
        if self.lm is not None:
            messagebox.showinfo("info", f"Loading model from {model_filename} successful!")
        else:
            messagebox.showerror("error", f"Failed to load model {model_filename}")
        pass

    def do_prediction(self):
        if self.lm is None:
            messagebox.showerror("error", "No model loaded!\nPlease train a model or load an existing model first.")
            return

        try:
            result = self.lm.predict([[self.area_income_value.get(),
                                       self.area_house_age_value.get(),
                                       self.area_number_of_rooms_value.get(),
                                       self.area_number_of_bedrooms_value.get(),
                                       self.area_population_value.get()]])
            self.prediction_price_value.set(result[0])
        except Exception as e:
            messagebox.showerror("error", f"Prediction failed: {str(e)}")
        pass
