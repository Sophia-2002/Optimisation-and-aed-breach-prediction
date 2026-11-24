"""
Interactive command-line management system integrating the scheduling model,
A&E analysis, breach prediction model and a data-management module with audit logging.

Run: python src/management_system.py   (AED4weeks.csv must be in the working directory)
"""
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import sys
import logging
from datetime import datetime

# ==========================================
#  Libraries for machine learning
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CLASS 1: Scheduler (Task 1 Logic)
# ==========================================
class Scheduler:
    def __init__(self):
        self.operators = ['E. Khan', 'Y. Chen', 'A. Taylor', 'R. Zidane', 'R. Perez', 'C. Santos']
        self.days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        self.wages = {'E. Khan': 25, 'Y. Chen': 26, 'A. Taylor': 24, 'R. Zidane': 23, 'R. Perez': 28, 'C. Santos': 30}
        
        self.availability = {
            'E. Khan': [6, 0, 6, 0, 6],
            'Y. Chen': [0, 6, 0, 6, 0],
            'A. Taylor': [4, 8, 4, 0, 4],
            'R. Zidane': [5, 5, 5, 0, 5],
            'R. Perez': [3, 0, 3, 8, 0],
            'C. Santos': [0, 0, 0, 6, 2]
        }
        
        self.min_hours = {
            'E. Khan': 8, 'Y. Chen': 8, 'A. Taylor': 8, 
            'R. Zidane': 8, 'R. Perez': 7, 'C. Santos': 7
        }
        
        self.status = None
        self.total_cost = 0.0
        self.schedule = None 
    
    def solve_model(self):
        """Run Linear Programming Model (Task 1)"""
        print("... Calculating optimal schedule ...")
        #initialize the LP minimization problem
        prob = pulp.LpProblem("Staff_Scheduling", pulp.LpMinimize)
        # Define Decision Variables: x[operator][day] = hours worked
        x = pulp.LpVariable.dicts("Hours", (self.operators, range(5)), lowBound=0, cat='Integer')
        # objective function: minimize total cost
        prob += pulp.lpSum([x[op][d] * self.wages[op] for op in self.operators for d in range(5)])
        #constraint 1: individual daily availability
        for op in self.operators:
            for d in range(5):
                prob += x[op][d] <= self.availability[op][d]
        #constraint2: daily coverage requirement        
        for d in range(5):
            prob += pulp.lpSum([x[op][d] for op in self.operators]) == 14
        #constraint 3: minimum weekly hours commitment per operator    
        for op in self.operators:
            prob += pulp.lpSum([x[op][d] for d in range(5)]) >= self.min_hours[op]
        #solve the problem    
        self.status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        #process results if optimal solution is found
        if pulp.LpStatus[self.status] == 'Optimal':
            self.total_cost = pulp.value(prob.objective)
            # format the schedule into a pandas dataframe for better readability
            data = {}
            for d_idx, day_name in enumerate(self.days):
                data[day_name] = [x[op][d_idx].varValue for op in self.operators]
            self.schedule = pd.DataFrame(data, index=self.operators)
            print("Optimization Successful.")
        else:
            print("No Solution Found.")
    #---helper methods for data access
    def get_total_cost(self): return self.total_cost
    def get_operator_hours(self, name):
        #return total assigned hours for a specific operator
        if self.schedule is not None and name in self.operators: return self.schedule.loc[name].sum()
        return 0
    def get_full_schedule(self): return self.schedule
    def get_day_schedule(self, day):
        #return the schedule for a specific day
        if self.schedule is not None and day in self.days: return self.schedule[day]
        return None
    def get_schedule_by_operator(self, name):
        #return the weekly schedule for a specific operator
        if self.schedule is not None and name in self.operators: return self.schedule.loc[name]
        return None

# ==========================================
# CLASS 2: AEDAnalyzer (Task 4, 5, 6 Logic)
# ==========================================
class AEDAnalyzer:
    def __init__(self, filepath='AED4weeks.csv'):
        self.filepath = filepath
        self.df = None
        self.sample_df = None
        
        # Set plotting theme
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
    def load_and_sample(self):
        """Load data, sample it, and perform necessary preprocessing for plots"""
        print(f"... Loading data from {self.filepath} ...")
        try:
            self.df = pd.read_csv(self.filepath)
            self.sample_df = self.df.sample(n=400, random_state=73701).copy() 
            
            # --- Preprocessing for Plots ---
            mapping = {'non-breach': 0, 'breach': 1, 0: 0, 1: 1}
            # Handle string formatting if necessary
            if self.sample_df['Breachornot'].dtype == 'object':
                 self.sample_df['Breachornot'] = self.sample_df['Breachornot'].str.strip().str.lower()
            
            self.sample_df['BreachNumeric'] = self.sample_df['Breachornot'].map(mapping)
            
            #create age groups
            age_bins = [0, 20, 30, 40, 50, 60, 70, 120]
            age_labels = ['Under 20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+']
            self.sample_df['AgeGroup'] = pd.cut(self.sample_df['Age'], bins=age_bins, labels=age_labels)
            
            print("Data loaded and preprocessed successfully.")
            
        except FileNotFoundError:
            print("Error: File not found.")
            self.df = None
            self.sample_df = None

    def show_plots(self):
        """interactive Menu to choose between Task 4 and Task 5 plots"""
        if self.sample_df is None:
            print("No data loaded.")
            return
            
        while True:
            print("\n" + "-"*30)
            print(" AED DATA VISUALIZATION MENU")
            print("-"*30)
            print("1. Descriptive Statistics (Task 4)")
            print("2. Breach Factor Analysis (Task 5)")
            print("0. Back to Main Menu")
            
            choice = input("Enter choice: ").strip().upper()
            
            if choice == '1':
                self._plot_task4_visuals()
            elif choice == '2':
                self._plot_task5_visuals()
            elif choice in ['0', 'Q', 'B']: 
                print("Returning to Main Menu...")
                break
            else:
                print("Invalid choice.")

    def _plot_task4_visuals(self):
        print("\nDisplaying Task 4 Plots... (Close each plot window to see the next one)")
        df = self.sample_df
        

        # Plot 1: Age
        plt.figure()
        plt.hist(np.asarray(df['Age'].dropna(), dtype=float), bins=20)
        plt.title('Age Distribution of Patients')
        plt.xlabel('Age')
        plt.show()


        # Plot 2: Day of Week
        plt.figure()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        existing_days = [d for d in dow_order if d in df['DayofWeek'].unique()]
        sns.countplot(x='DayofWeek', data=df, order=existing_days, palette='viridis')
        plt.title('Patient Arrivals by Day of the Week')
        plt.xticks(rotation=45)
        plt.show()

        # Plot 3: Breach proportion(Pie Chart)
        plt.figure()
        breach_counts = df['BreachNumeric'].value_counts()
        labels = ['Not Breach (0)', 'Breached (1)'] if len(breach_counts) == 2 else breach_counts.index
        plt.pie(breach_counts, labels=labels, autopct='%1.1f%%', 
                colors=['lightgreen', 'salmon'], startangle=140, explode=[0.05]*len(breach_counts))
        plt.title('Proportion of 4-Hour Target Breaches')
        plt.show()

        # Plot 4: Top 10 HRG Categories
        plt.figure()
        top_hrg = df['HRG'].value_counts().nlargest(10)
        sns.barplot(x=top_hrg.values, y=top_hrg.index, palette='magma')
        plt.title('Top 10 HRG Categories')
        plt.xlabel('Number of Patients')
        plt.show()


        # Plot 5: Peak Hours Analysis
        plt.figure()
        peak_hours = df.groupby('Period').size().sort_index()
        plt.plot(np.asarray(peak_hours.index),np.asarray(peak_hours.values),
                 marker='o',linestyle='-',color="blue")
        plt.title('Peak Hour Analysis (Arrivals)')
        plt.xlabel('Hour of Day (Period)')
        plt.ylabel('Number of Patients')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Plot 6: Correlation Matrix
        plt.figure(figsize=(10, 8))
        numeric_cols = ['Age', 'Period', 'LoS', 'noofinvestigation', 'nooftreatment', 'noofpatients', 'BreachNumeric']
        cols_to_use = [c for c in numeric_cols if c in df.columns]
        corr = df[cols_to_use].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

        # Plot 7: Length of stay vs workload
        plt.figure()
        sns.scatterplot(x='noofpatients', y='LoS', hue='BreachNumeric', data=df, palette='coolwarm', alpha=0.6)
        plt.axhline(y=240, color='red', linestyle='--', label='Target (240 mins)')
        plt.title('Length of Stay vs. Workload at Arrival')
        plt.legend(title='Breach (1=Yes)')
        plt.show()

        # Plot 8: distribution of length of stay
        plt.figure()
        sns.kdeplot(df['LoS'], fill=True, color='purple')
        plt.axvline(x=240, color='red', linestyle='--', label='Target')
        plt.title('Distribution of Length of Stay')
        plt.show()

    def _plot_task5_visuals(self):
        print("\nDisplaying Task 5 Plots... (Close each plot window to see the next one)")
        df = self.sample_df
        
        # Plot 1 :Los by breacch status (boxplot)
        plt.figure()
        sns.boxplot(x='Breachornot', y='LoS', data=df, palette="Set2")
        plt.axhline(y=240, color='red', linestyle='--', label='Target')
        plt.title('LoS Distribution by Breach Status')
        plt.show()



        # Plot 2 breach rate by period
        period_breach = df.groupby('Period')['BreachNumeric'].mean() * 100
        plt.figure()
        plt.plot(np.asarray(period_breach.index),np.asarray(period_breach.values),
                 marker='o',linestyle='-')
        plt.title('Breach Rate by Arrival Time')
        plt.ylabel('Breach Rate (%)')
        plt.grid(True)
        plt.show()



        # Plot 3 breach rate by age group
        age_trend = df.groupby('AgeGroup', observed=False)['BreachNumeric'].mean() * 100
        plt.figure()
        sns.barplot(x=age_trend.index, y=age_trend.values, palette="Blues_d")
        plt.title('Breach Rate by Age Group')
        plt.ylabel('Breach Rate (%)')
        plt.xticks(rotation=45)
        plt.show()


        # Plot 4 breach rate vs Investigation
        invest_trend = df.groupby('noofinvestigation')['BreachNumeric'].mean() * 100
        plt.figure()
        plt.plot(
            np.asarray(invest_trend.index),
            np.asarray(invest_trend.values),
            marker='s',
            linestyle='-',color='darkred'
            )
        plt.title('Breach Rate vs. No. of Investigations')
        plt.ylabel('Breach Rate (%)')
        plt.grid(True)
        plt.show()
        

        # Plot 5 breach rate vs treatments
        treat_trend = df.groupby('nooftreatment')['BreachNumeric'].mean() * 100
        plt.figure()
        plt.plot(treat_trend.index.to_numpy(),treat_trend.to_numpy(),
                 marker='^',linestyle='-',color="green")
        plt.title('Breach Rate vs. No. of Treatments')
        plt.ylabel('Breach Rate (%)')
        plt.grid(True)
        plt.show()

    def run_prediction_model(self):
        """Task 6: Logistic Regression Analysis"""
        if self.df is None:
            print("Data not loaded. Cannot run model.")
            return

        print("\n>>> Starting Task 6: Logistic Regression Model...")
        df_model = self.df.copy()
        #Define target and features
        target = 'Breachornot'
        numeric_features = ['Age', 'noofpatients', 'noofinvestigation', 'nooftreatment']
        categorical_features = ['HRG', 'DayofWeek', 'Period']
        
        #check for missing columns
        missing = [c for c in numeric_features + categorical_features + [target] if c not in df_model.columns]
        if missing:
            print(f"Error: Missing columns: {missing}")
            return
        #prepare x and y
        X = df_model[numeric_features + categorical_features]
        df_model[target] = df_model[target].astype(str).str.lower().str.strip()
        y = df_model[target].map({'non-breach': 0, 'breach': 1})
        #drop rows with missing target values
        if y.isnull().any():
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
        #define preprocessor: scale numeric data, one-hot encode category
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder( handle_unknown='ignore'), categorical_features) ###這裡有改
            ]
        )
        #apply transformations
        try:
            X_processed = preprocessor.fit_transform(X)
        except Exception as e:
            print(f"Preprocessing Error: {e}")
            return
        #split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42, stratify=y)
        #train logistic regression model (with balanced class weights)
        logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
        logreg.fit(X_train, y_train)
        y_pred = (logreg.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        
        #output 1:classification report
        print("\n=== Logistic Regression (Threshold = 0.5) ===")
        print(classification_report(y_test, y_pred))
        
        #output 2:confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
        #output3: odds ratios (feature importance)
        try:
            ohe = preprocessor.named_transformers_['cat']
            cat_names = ohe.get_feature_names_out(categorical_features)
            feats = numeric_features + list(cat_names)

            if len(logreg.coef_[0]) == len(feats):
                coef_df = pd.DataFrame({'Feature': feats, 'Odds_Ratio': np.exp(logreg.coef_[0])})
                coef_df = coef_df.sort_values(by='Odds_Ratio', ascending=False)
                
                print("\n=== Logistic Regression Odds Ratios ===")
                print(coef_df.to_string(index=False))
        except Exception as e:
            print(f"Error showing Odds Ratios: {e}")


# ==========================================
# CLASS 3: AEDDataManager (Task 8 Logic)
# ==========================================
class AEDDataManager:
    """
    Task 8: Interactive data management + audit logging.
    """
    def __init__(self, filepath="AED4weeks.csv", log_path="user_actions.log"):
        self.filepath = filepath
        self.log_path = log_path
        self.df = None

        # logger (avoid duplicate handlers on rerun)
        self.logger = logging.getLogger("AED_TASK8_AUDIT")
        self.logger.setLevel(logging.INFO)
        # prevent duplicate handlers if re-run in juptyter
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def log(self, action: str):
        #write an action to the log file
        self.logger.info(action)

    def load(self):
        #load the csv file into a dataframe
        self.df = pd.read_csv(self.filepath)
        self.log(f"LOAD | file={self.filepath} | rows={len(self.df)}")

    def ensure_loaded(self):
        #ensure data is loaded before performing operations.
        if self.df is None:
            self.load()

    # Function 1) Lookup by ID
    def lookup_by_id(self, patient_id: str):
        self.ensure_loaded()
        res = self.df[self.df["ID"] == patient_id]
        self.log(f"LOOKUP_ID | ID={patient_id} | found={len(res)}")
        return res

    # Function 2) Filter by numeric range
    def filter_range(self, column: str, min_val=None, max_val=None) -> pd.DataFrame:
        self.ensure_loaded()
        if column not in self.df.columns:
            self.log(f"FILTER_FAIL | column={column} | reason=column_not_found")
            raise KeyError(f"Column '{column}' not found.")
        #convert column to numeric, coercing errors
        s = pd.to_numeric(self.df[column], errors="coerce")
        mask = s.notna()
        if min_val is not None:
            mask &= (s >= min_val)
        if max_val is not None:
            mask &= (s <= max_val)

        out = self.df[mask].copy()
        self.log(f"FILTER_RANGE | column={column} | min={min_val} | max={max_val} | rows={len(out)}")
        return out

    # Function 3) Modify a record (field value)
    def modify(self, patient_id: int, field: str, new_value: str) -> bool:
        self.ensure_loaded()
        if field not in self.df.columns:
            self.log(f"MODIFY_FAIL | ID={patient_id} | field={field} | reason=field_not_found")
            raise KeyError(f"Field '{field}' not found.")

        idx_list = self.df.index[self.df["ID"] == patient_id].tolist()
        if not idx_list:
            self.log(f"MODIFY_FAIL | ID={patient_id} | field={field} | reason=id_not_found")
            return False

        i = idx_list[0]
        old = self.df.at[i, field]
        #update value
        self.df.at[i, field] = new_value
        self.log(f"MODIFY | ID={patient_id} | field={field} | old={old} | new={new_value}")
        return True

    # Function 4) Delete a patient record
    def delete(self, patient_id: int) -> int:
        self.ensure_loaded()
        before = len(self.df)
        self.df = self.df[self.df["ID"] != patient_id].copy()
        deleted = before - len(self.df)
        self.log(f"DELETE | ID={patient_id} | deleted={deleted}")
        return deleted

    # Function 5) Save changes to a new CSV file
    def save(self, out_path="AED4weeks_modified.csv") -> str:
        self.ensure_loaded()
        self.df.to_csv(out_path, index=False)
        self.log(f"SAVE | out={out_path} | rows={len(self.df)}")
        return out_path
    
    # Function 6) Add a new patient record (NEW FUNCTION)
    def add_record(self, record_dict: dict) -> bool:
        self.ensure_loaded()
        new_id = record_dict.get("ID")
        
        # Check if ID already exists to prevent duplicates
        if new_id in self.df["ID"].values:
            self.log(f"ADD_FAIL | ID={new_id} | reason=duplicate_id")
            return False
            
        # Create a DataFrame for the new row and append it
        new_row = pd.DataFrame([record_dict])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        self.log(f"ADD | ID={new_id} | record={record_dict}")
        return True


# ==========================================
# 4. Helper Functions
# ==========================================

def get_user_selection(options, prompt_msg):
    """Helper function to list options and get a valid numeric selection."""
    while True:
        print(f"\n{prompt_msg}")
        for i, item in enumerate(options, 1):
            print(f"{i}: {item}")
        print("0: Back to Main Menu")
    
        try:
            user_input = input("> Enter choice number (or 0 to back): ").strip().upper()
            
            if user_input in ['0', 'Q', 'B']:
                return None
            
            idx = int(user_input) - 1
            if 0 <= idx < len(options):
                return options[idx]
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

            
def select_from_list(options, prompt_text):
    """
    Helper function: Forces user to select a valid option from a list.
    Returns the string value of the selected option.
    """
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        print(f"   {i}. {opt}")
    
    while True:
        choice = input(f"   > Select an option (1-{len(options)}): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("   [!] Invalid selection. Please enter a valid number.")

def task8_menu(manager):
    """Sub-menu for Task 8 Data Management"""
    manager.ensure_loaded()

    while True:
        print("\n" + "="*45)
        print("Task 8 – Data Management Menu")
        print("1: Lookup patient by ID")
        print("2: Filter patients by variable range (min/max)")
        print("3: Modify a patient's field")
        print("4: Delete a patient by ID")
        print("5: Save current data to CSV")
        print("6: Add a new patient")
        print("B: Back to Main Menu")
        print("="*45)

        c = input("Choose an option: ").strip().upper()

        if c == "1":
            pid = input("Enter patient ID (e.g., P10000): ").strip().upper()
            if not pid.startswith("P"):
                print("Patient ID must start with 'P' (e.g., P10000).")
                continue

            res = manager.lookup_by_id(pid)
            if res.empty:
                print("No patient found.")
            else:
                print(res.to_string(index=False))

        elif c == "2":
            col = input("Enter column name (e.g., Age, LoS, noofpatients): ").strip()
            mn = input("Min (blank = none): ").strip()
            mx = input("Max (blank = none): ").strip()

            mn = float(mn) if mn != "" else None
            mx = float(mx) if mx != "" else None

            try:
                out = manager.filter_range(col, mn, mx)
                print(f"Found {len(out)} patients.")
                print(out.head(20).to_string(index=False))
                if len(out) > 20:
                    print("(Showing first 20 rows)")
            except Exception as e:
                print(f"Error: {e}")

        elif c == "3":
            pid = input("Enter patient ID (e.g., P10000): ").strip().upper()
            if not pid.startswith("P"):
                print("Patient ID must start with 'P' (e.g., P10000).")
                continue

            field = input("Enter field to modify: ").strip()
            newv = input("Enter new value: ").strip()

            try:
                ok = manager.modify(pid, field, newv)
                print("Updated." if ok else "Patient ID not found.")
            except Exception as e:
                print(f"Error: {e}")

        elif c == "4":
            pid = input("Enter patient ID (e.g., P10000) to delete: ").strip().upper()
            if not pid.startswith("P"):
                print("Patient ID must start with 'P' (e.g., P10000).")
                continue

            confirm = input("Type DELETE to confirm: ").strip().upper()
            if confirm != "DELETE":
                print("Cancelled.")
                continue

            deleted = manager.delete(pid)
            print(f"Deleted rows: {deleted}")

        elif c == "5":
            out = manager.save()
            print(f"Saved to: {out}")
            
        elif c == "6":
            print("\n--- Add New Patient Record (Strict Validation) ---")
            
            # 1. Get ID
            pid = input("1. Enter new patient ID (e.g., P99999): ").strip().upper()
            if not pid.startswith("P"):
                print("   [!] ID must start with 'P'. Action cancelled.")
                continue
            
            try:
                # 2. Age (Numeric)
                while True:
                    age_in = input("2. Enter Age (integer): ").strip()
                    if age_in.isdigit():
                        age = int(age_in)
                        break
                    print("   [!] Please enter a valid integer for Age.")

                # 3. Length of Stay (Numeric)
                while True:
                    los_in = input("3. Enter Length of Stay (LoS) (float): ").strip()
                    try:
                        los = float(los_in)
                        break
                    except ValueError:
                        print("   [!] Please enter a valid number for LoS.")

                # 4. Breach Status (Selection)
                # Define valid options
                breach_opts = ["breach", "non-breach"]
                breach = select_from_list(breach_opts, "4. Select Breach Status:")

                # 5. HRG Code (Selection from existing data)
               # 1） Fetch existing codes from the current database
                if manager.df is not None and 'HRG' in manager.df.columns:
                    current_hrgs = sorted([str(x) for x in manager.df['HRG'].dropna().unique().tolist()])
                else:
                    current_hrgs = []

                # 2） Add "Other" option to allow users to enter a new code
                hrg_opts = current_hrgs + ["Other (Enter new code)"]
                
                print(f"\n5. Select HRG Code:")
                # Use the previously defined helper function to display the menu
                selected_opt = select_from_list(hrg_opts, "Choose from existing or enter new:")

                # 3） Check the user's selection
                if selected_opt == "Other (Enter new code)":
                    while True:
                        custom_hrg = input("   > Enter New HRG Code: ").strip()
                        if custom_hrg:  # Ensure it is not an empty string
                            hrg = custom_hrg
                            break
                        print("   [!] HRG Code cannot be empty.")
                else:
                    hrg = selected_opt
                # 6. Day of Week (Selection)
                dow_opts = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                dow = select_from_list(dow_opts, "6. Select Day of Week:")

                # 7. Arrival Period (Numeric 0-23)
                while True:
                    period_in = input("7. Enter Arrival Period (hour 0-23): ").strip()
                    if period_in.isdigit() and 0 <= int(period_in) <= 23:
                        period = int(period_in)
                        break
                    print("   [!] Please enter a valid hour between 0 and 23.")
                
                # 8. Investigations (Numeric)
                inv_in = input("8. Enter No. of Investigations (default 0): ").strip()
                n_inv = int(inv_in) if inv_in.isdigit() else 0
                
                # 9. Treatments (Numeric)
                treat_in = input("9. Enter No. of Treatments (default 0): ").strip()
                n_treat = int(treat_in) if treat_in.isdigit() else 0
                
                # 10. Patient Count (Numeric)
                pat_in = input("10. Enter No. of Patients in dept (default 0): ").strip()
                n_pat = int(pat_in) if pat_in.isdigit() else 0

                # Construct the dictionary record
                new_record = {
                    "ID": pid,
                    "Age": age,
                    "LoS": los,
                    "Breachornot": breach,
                    "HRG": hrg,
                    "DayofWeek": dow,
                    "Period": period,
                    "noofinvestigation": n_inv,
                    "nooftreatment": n_treat,
                    "noofpatients": n_pat
                }
                
                # Call manager to add the record
                success = manager.add_record(new_record)
                
                if success:
                    print(f"\n[Success] Patient {pid} added to the dataset.")
                else:
                    print(f"\n[Error] Patient ID {pid} already exists.")
                    
            except Exception as e:
                print(f"\n[Error] An unexpected error occurred: {e}")
        elif c == "B":
            print("Returning to Main Menu...")
            break

        else:
            print("Invalid option.")

# ==========================================
# 5. Main Interface (Unified)
# ==========================================
def main():
    print("Initializing System...")
    
    # Initialize Task 1: schedule optimization
    scheduler = Scheduler()
    scheduler.solve_model()
    
    # Initialize Task 4-6 data analysis & ML
    analyzer = AEDAnalyzer('AED4weeks.csv')
    analyzer.load_and_sample()
    
    # Initialize Task 8: Data management & logging
    # Initialize the manager object, ready to handle data management requests
    data_manager = AEDDataManager("AED4weeks.csv", log_path="user_actions.log")
    
    while True:
        print("\n" + "="*40)
        print(" MANCHESTER HOSPITAL MANAGEMENT SYSTEM")
        print("="*40)
        print("1: Show overall cost")
        print("2: Show hours assigned to an operator")
        print("3: Show schedule on a particular day")
        print("4: Show schedule of an operator")
        print("5: Show overall schedule table")
        print("6: Show AED Data Plots (Descriptive/Factors)")
        print("7: Run Breach Prediction Model (ML)")
        print("8: Data Management")
        print("Q: Quit")
        print("-" * 40)
        
        choice = input("Choose an option: ").strip().upper()
        
        if choice == '1':
            print(f"> Total Cost: £{scheduler.get_total_cost():.2f}")
            
        elif choice == '2':
            while True:
                name = get_user_selection(scheduler.operators, "Select an operator to check hours:")
                if name is None:
                    print("--- Returning to Main Menu ---")
                    break
                print(f"> {name} is assigned {scheduler.get_operator_hours(name):.1f} hours.")
            
        elif choice == '3':
            while True:
                day = get_user_selection(scheduler.days, "Select a day to check schedule:")
                if day is None:
                    print("--- Returning to Main Menu ---")
                    break
                print(f"\n--- Schedule for {day} ---")
                print(scheduler.get_day_schedule(day))
            
        elif choice == '4':
            while True:
                name = get_user_selection(scheduler.operators, "Select an operator to see full schedule:")
                if name is None:
                    print("--- Returning to Main Menu ---")
                    break
                print(f"\n--- Schedule for {name} ---")
                print(scheduler.get_schedule_by_operator(name))
            
        elif choice == '5':
            print("\n--- Overall Schedule ---")
            print(scheduler.get_full_schedule())
            
        elif choice == '6':
            # Calls the submenu inside AEDAnalyzer
            analyzer.show_plots()
            
        elif choice == '7':
            analyzer.run_prediction_model()
            
        elif choice == "8":
            # Call the task8_menu function defined above, passing in the data_manager
            task8_menu(data_manager)
            
        elif choice == 'Q':
            print("Good bye!")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()