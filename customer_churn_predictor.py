import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Loading the Dataset
df = pd.read_csv('Customer Churn.csv')

# displaying the first 10 rows
print(df. head(10))
# Displaying the Shape of the Data set (3150 rows, 14cols)
print("DataShape:", df.shape)

# Task 2: Tasks using Pandas
# a) Search operation: Customers with high service call volume using the 'frequency of use'coln

high_service_calls = df.query('`Frequency of use` > 100')

print ("Customers with high service call volume:", len(high_service_calls))

# outputing Customers with high service call volume to a csv file

high_service_calls.to_csv('high_service_calls.csv', index=False)

# b) Sort Operation: Sort by Tenure(Subscription length) and Charges to help identify whether the 2 factors are associated with customer retention.

df_sorted = df.sort_values(by=['Subscription  Length', 'Charge  Amount'], ascending=[False, False])

# outputing the sorted dataset to a csv file 
# df_sorted.to_csv('sorted_dataset.csv', index=False)


# c) Iteration operation: for age-group with the most customer value using a for loop

age_group_values = {}

for index, row in df.iterrows():

    age_group = row['Age Group']
    customer_value = row['Customer Value']

    # Check whether age group already exists
    if age_group in age_group_values:

        # Add customer value to existing total
        age_group_values[age_group] += customer_value

    else:

        # Create a new age group and assign its value
        age_group_values[age_group] = customer_value

print("TOTAL CUSTOMER VALUE BY AGE GROUP")


for age_group, total_value in sorted(age_group_values.items()):

    print(f"Age Group {age_group}: {total_value:.2f}")

# Age group with the highest customer value
highest_value_age_group = max(
    age_group_values,
    key=age_group_values.get
)

print("\nHIGHEST VALUE AGE GROUP:", highest_value_age_group)

print(
    "Total Customer Value:",
    round(age_group_values[highest_value_age_group], 2)
)


# Datascience Analysis using Pandas and Matplotlib

# 1) Retained vs Churned Customers Distribution
churn_counts = df['Churn'].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    ['Retained', 'Churned'],
    churn_counts.sort_index(),
)

plt.title('Customer Churn Distribution')
plt.xlabel('Customer Status')
plt.ylabel('Number of Customers')

plt.savefig('viz/Customer_Churn_Distribution.jpg',format='jpg',dpi=300,bbox_inches='tight')

plt.show()

# 2) Customer Value Distribution by Age Group
age_group_value = df.groupby('Age Group')['Customer Value'].sum()
plt.figure(figsize=(8, 5))

plt.bar(
    age_group_value.index,
    age_group_value.values
)

plt.title('Total Customer Value by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Customer Value')
plt.xticks(age_group_value.index)
plt.savefig('viz/Customer_Value_by_Age_Group.jpg',
            format='jpg',
            dpi=300,
            bbox_inches='tight')

plt.show()