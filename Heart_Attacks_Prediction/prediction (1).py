

def fc(dataFrame):
    from pyspark.sql import SparkSession,SQLContext
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import StringIndexer
    
    sc = SparkSession.builder.getOrCreate() 
    sq = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)
    df = sq.read.csv("transformed_heart_2022.csv", inferSchema ='true' ,header='true',sep=',')

    dataFrame = sc.createDataFrame(dataFrame)
    
    
    input=["sex","generalHealth","PhysicalHealthDays","physicalActivities","removedTeeth","hadAngina","hadHeartAttack","hadStroke",
           "hadCOPD","hadKidneyDisease","hadArthritis","hadDiabetes",
           "deafOrHardOfHearing","blindOrVisionDifficulty","difficultyWalking",
           "difficultyDressingBathing","difficultyErrands","smokerStatus","chestScan",
                  "ageCategory","alcoholDrinkers","pneumoVaxEver"]
    output = ["Sex","GeneralHealth","physicalHealthDays","PhysicalActivities",
              "RemovedTeeth","HadAngina","HadHeartAttack","HadStroke","HadCOPD",
              "HadKidneyDisease","HadArthritis",
                 "HadDiabetes","DeafOrHardOfHearing","BlindOrVisionDifficulty","DifficultyWalking",
              "DifficultyDressingBathing","DifficultyErrands","SmokerStatus","ChestScan",
                  "AgeCategory","AlcoholDrinkers","PneumoVaxEver"]

    
    first_in =["sex","generalHealth","PhysicalHealthDays","physicalActivities","removedTeeth","hadAngina","hadStroke",
           "hadCOPD","hadKidneyDisease","hadArthritis","hadDiabetes",
           "deafOrHardOfHearing","blindOrVisionDifficulty","difficultyWalking",
           "difficultyDressingBathing","difficultyErrands","smokerStatus","chestScan",
                  "ageCategory","alcoholDrinkers","pneumoVaxEver"]
    second_in = ["Sex","GeneralHealth","physicalHealthDays","PhysicalActivities",
              "RemovedTeeth","HadAngina","HadStroke","HadCOPD",
              "HadKidneyDisease","HadArthritis",
                 "HadDiabetes","DeafOrHardOfHearing","BlindOrVisionDifficulty","DifficultyWalking",
              "DifficultyDressingBathing","DifficultyErrands","SmokerStatus","ChestScan",
                  "AgeCategory","AlcoholDrinkers","PneumoVaxEver"]


    

    encoder = StringIndexer(inputCols=input,outputCols=output)
    df_encoded = encoder.fit(df).transform(df)

    encoder1 = StringIndexer(inputCols=first_in,outputCols=second_in)
    df_encoded1 = encoder1.fit(dataFrame).transform(dataFrame)

    

    
    features_cls = output = ["Sex","GeneralHealth","physicalHealthDays","PhysicalActivities",
                             "RemovedTeeth","HadAngina","HadStroke","HadCOPD","HadKidneyDisease","HadArthritis",
                 "HadDiabetes","DeafOrHardOfHearing","BlindOrVisionDifficulty",
                             "DifficultyWalking","DifficultyDressingBathing","DifficultyErrands","SmokerStatus","ChestScan",
                  "AgeCategory","AlcoholDrinkers","PneumoVaxEver"]

    
    vector_assembler = VectorAssembler(inputCols=features_cls, outputCol="features")
    df_encoded = vector_assembler.transform(df_encoded)
    df_encoded1 = vector_assembler.transform(df_encoded1)

    
    train_data, test_data = df_encoded.randomSplit([0.8, 0.2], seed=123)
    dt = DecisionTreeClassifier(labelCol="HadHeartAttack", featuresCol="features")

    model = dt.fit(train_data)
    
    predictions = model.transform(df_encoded1)

    pred = predictions.select("prediction")
    first_row = pred.first()


    prediction_value = str(first_row.prediction)
    

    if prediction_value == "1.0":
        return "This patient would experience a heart attack"
    else: return "This patient is not likely to experience a heart attack"
    
        

    
    


