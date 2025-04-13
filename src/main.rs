use dataframe_project::{DataFrame, Value, ValueType, Error};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. Create DataFrame from CSV ---
    println!("--- Reading data.csv ---");
    let mut col_types = HashMap::new();
    col_types.insert("Name".to_string(), ValueType::String);
    col_types.insert("Number".to_string(), ValueType::I64);
    col_types.insert("PPG".to_string(), ValueType::F64);
    col_types.insert("YearBorn".to_string(), ValueType::I64);
    col_types.insert("TotalPoints".to_string(), ValueType::I64);
    col_types.insert("LikesPizza".to_string(), ValueType::Bool);

    let mut df1 = DataFrame::read_csv("data.csv", &col_types)?;
    println!("Initial DataFrame (df1):");
    println!("{}", df1);

    // --- Add Column --- 
    println!("\n--- Adding 'hall_of_fame' column ---");
    let hall_of_fame_data = vec![
        Value::Bool(true),  // Kareem
        Value::Bool(true),  // Karl
        Value::Bool(true),  // LeBron
        Value::Bool(true),  // Kobe
        Value::Bool(true),  // Michael
        Value::Bool(true),  // Shaq
        Value::Bool(true),  // Wilt
        Value::Bool(true),  // Dirk
        Value::Bool(true),  // Tim
        Value::Bool(true),  // Hakeem
    ];
    df1 = df1.add_column("hall_of_fame".to_string(), hall_of_fame_data)?;
    println!("DataFrame after adding column:");
    println!("{}", df1);

    // --- Merge Frame --- 
    println!("\n--- Reading data2.csv and merging ---");
    // Reuse col_types, assuming data2.csv has the same structure initially
    let df2 = DataFrame::read_csv("data2.csv", &col_types)?;
    println!("Second DataFrame (df2):");
    println!("{}", df2);

    // Need to add the 'hall_of_fame' column to df2 before merging
    let hall_of_fame_data2 = vec![
        Value::Bool(true), // Magic
        Value::Bool(true), // Larry
        Value::Bool(true), // Bill
        Value::Bool(true), // Oscar
        Value::Bool(true), // Jerry
        Value::Bool(true), // David
    ]; 
    let df2_with_hof = df2.add_column("hall_of_fame".to_string(), hall_of_fame_data2)?;

    let merged_df = df1.merge_frame(&df2_with_hof)?;
    println!("Merged DataFrame:");
    println!("{}", merged_df);

    // --- Restrict Columns --- 
    println!("\n--- Restricting columns to 'Name' and 'TotalPoints' ---");
    let restricted_df = merged_df.restrict_columns(&["Name".to_string(), "TotalPoints".to_string()])?;
    println!("Restricted DataFrame:");
    println!("{}", restricted_df);

    // --- Filter --- 
    println!("\n--- Filtering for players with PPG > 25.0 ---");
    let filtered_df = merged_df.filter("PPG", |value| {
        match value {
            Value::F64(ppg) => *ppg > 25.0,
            _ => false, // Only filter based on F64 PPG values
        }
    })?;
    println!("Filtered DataFrame (PPG > 25.0):");
    println!("{}", filtered_df);

    // --- Column Op: Median --- 
    println!("\n--- Calculating Median PPG ---");
    // Use the merged dataframe for median calculation
    let median_ppg = merged_df.median("PPG")?;
    println!("Median PPG: {}", median_ppg);

    // --- Column Op: Sub Columns --- 
    println!("\n--- Subtracting YearBorn from TotalPoints ---");
    // Use the merged dataframe for subtraction
    let diff_vec = merged_df.sub_columns("TotalPoints", "YearBorn")?;
    println!("Result of TotalPoints - YearBorn:");
    // Print the resulting vector nicely
    let diff_strings: Vec<String> = diff_vec.iter().map(|v| v.to_string()).collect();
    println!("[ {} ]", diff_strings.join(", "));

    // Example of column_op for a custom operation (e.g., sum of a column)
    println!("\n--- Using column_op to calculate sum of TotalPoints ---");
    let total_points_sum = merged_df.column_op(&["TotalPoints".to_string()], |cols| {
        let col = cols[0];
        let mut sum: i64 = 0;
        for val in &col.data {
            if let Value::I64(points) = val {
                sum += points;
            } else {
                return Err(Error::TypeMismatch("Expected I64 for TotalPoints sum".to_string()));
            }
        }
        Ok(Value::I64(sum))
    })?;
    println!("Sum of TotalPoints (calculated via column_op): {}", total_points_sum);


    Ok(())
} 