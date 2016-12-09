/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2016 HPCC Systems®.  All rights reserved.
############################################################################## */

IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT PBblas;
IMPORT PBblas.Types as pbbTypes;
IMPORT PBblas.Converted as pbbConverted;
IMPORT PBblas.MatUtils;

NotCompat := PBblas.Constants.Dimension_Incompat;
NumericField := Types.NumericField;
Layout_Cell := pbbTypes.Layout_Cell;
Layout_Model := Types.Layout_Model;
t_work_item  := Types.t_work_item;
t_RecordID   := Types.t_RecordID;
t_FieldNumber := Types.t_FieldNumber;
t_FieldReal   := Types.t_FieldReal;
t_Discrete    := Types.t_Discrete;
t_Count       := Types.t_Count;
empty_data := DATASET([], NumericField);
triangle := pbbTypes.triangle;
side := pbbTypes.side;
diagonal := pbbTypes.diagonal;
/**
  * Ordinary Least Squares (OLS) Linear Regression
  *  aka Ordinary Linear Regression
  *
  * Regression learns a function that maps a set of input data (independents)
  * to one or more output variables (dependents).  The resulting learned function is
  * known as the model.  That model can then be used repetitively to predict 
  * (i.e. estimate) the output value(s) based on new input data.
  *
  * Two major use cases are supported:
  * 1) Learn and return a model
  * 2) Use an existing (e.g. persisted) model to predict new values for Y
  *
  * Of course, both can be done in a single run.  Alternatively, the
  * model can be persisted and used indefinitely for prediction of Y values,
  * as long as the record format has not changed, and the original training
  * data remains representative of the population.  
  *
  * OLS supports any number of independent variables and a single dependent 
  * variable.  Y must therefore be specified as a vector (i.e. single field,
  * 'number'=1).
  *
  * Training data is presented as parameters to this module.  When using a
  * previously persisted model (use case 2 above), these parameters should
  * be omitted.
  * 
  * @param X The independent training data in DATASET(NumericField) format.
  *          Each observation (e.g. record) is identified by
  *          'id', and each feature is identified by field number (i.e. 
  *          'number').  Omit this parameter when predicting from a persisted
  *          model.
  *           
  * @param Y The dependent variable training data in DATASET(NumericField)
  *         format Each observation (e.g. record) is identified by
  *          'id', and the field number should be 1). Omit this parameter when 
  *         predicting from a persisted model.
  *         
  */
EXPORT OLS(DATASET(NumericField) X=empty_data,
		   DATASET(NumericField) Y=empty_data)
                     := MODULE(ML_Core.Interfaces.IRegression())
  // Convert X to matrix form
  mX0 := pbbConverted.NFToMatrix(X);
  // Insert column of ones for Y intercept
  mX := MatUtils.InsertCols(mX0, 1, 1.0);
  // Convert Y to matrix form
  mY := pbbConverted.NFToMatrix(Y);
  DATASET(Layout_Cell) learnBetas := FUNCTION
    XtX := PBblas.gemm(TRUE, FALSE, 1.0, mX, mX);
    XtY := PBblas.gemm(TRUE, FALSE, 1.0, mX, mY);
    L   := PBblas.potrf(triangle.Lower, XtX);
    s1  := PBblas.trsm(Side.Ax, triangle.Lower, FALSE, diagonal.NotUnitTri, 1.0,
                          L, XtY);
    B:= PBblas.trsm(Side.Ax, triangle.Upper, TRUE, diagonal.NotUnitTri, 1.0,
                          L, s1);
    return B;
  END;
  mBetas := learnBetas;
  Layout_Model betasToModel(Layout_Cell b) := TRANSFORM
    SELF.wi := b.wi_id;
    SELF.id := 1;
    SELF.number    := b.x;
    SELF.value     := b.v;
  END;
  /**
    * GetModel
    *
    * Returns the learned model that maps X's to Y's.  In the case of
    * OLS, the model represents a set of Betas which are the coefficients
    * of the linear model: Beta0 * 1 + Beta1 * Field1 + Beta2 * Field2 ...
    * The ID of each model record is 1 since betas are the only type of
    * data in the model.
    * The Field Number ('number') indicates to which field of X the beta is to be
    * applied.  Field number 0 provides the intercept portion of the linear
    * model and is always multiplied by 1.
    * Note that if multiple work-items are provided within X and Y, there
    * will be multiple models returned.  The models can be separated by
    * their work item id (i.e. 'wi).  A single model can be extracted from
    * a myriad model by using e.g., model(wi=myWI_id).  GetModel should not
    * be called when predicting using a previously persisted model (i.e. when
    * training data was not passed to the module.
    *
    * @return  Model in DATASET(Layout_Model) format
    * @see     ML_core/Types.Layout_Model
    */
  EXPORT DATASET(Layout_Model) GetModel  := PROJECT(mBetas, betasToModel(LEFT));
  
  // Get betas from the model
  // Convert model to a row vector of betas
  Layout_Cell modelToBetas(Layout_Model m) := TRANSFORM
	SELF.wi_id := m.wi;
	SELF.x     := m.number;
	SELF.y     := 1;
	SELF.v     := m.value;
  END;
  SHARED DATASET(Layout_Cell) mBetas(DATASET(Layout_Model) model=GetModel) := 
                                                PROJECT(model, modelToBetas(LEFT));
  /**
    * Return raw Beta values as numeric fields
    *
    * Extracts Beta values from the model.  Can be used during training and prediction 
    * phases.  For use during training phase, the 'model' parameter can be omitted.
    * GetModel will be called to retrieve the model based on the training data.  For
    * use during prediction phase, a previously persisted model should be provided.
    * The 'id' field of the returned NumericField records is not used and is set to 1.
    * The 'number' field of the returned record indicates the position of the Beta
    * value.  1 provides the Beta for the constant term (i.e. the Y intercept) while
    * subsequent values reflect the Beta for each correspondingly numbered X feature.
    * Feature 1 corresponds to Beta with 'number' = 2 and so on.
    * If 'model' contains multiple work-items, Separate sets of Betas will be returned
    * for each of the 'myriad' models (distinguished by 'wi').
    *
    * @param model Optional parameter provides a model that was previously retrieved
    *               using GetModel.  If omitted, GetModel will be used as the model.
    * @return DATASET(NumericField) containing the Beta values.
    * 
    */
  EXPORT DATASET(NumericField) Betas(DATASET(Layout_Model) model=GetModel) :=
                                                PbbConverted.MatrixToNF(mBetas(model));
  /**
    * Predict the dependent variable values (Y) for any set of independent
    * variables (X).  Returns a predicted Y value for each observation
    * (i.e. record) of X.
    * 
    * This supports the 'myriad' style interface in that multiple independent
    * work items may be present in 'newX', and multiple independent models may
    * be provided in 'model'.  The resulting predicted values will also be
    * separable by work item (i.e. wi).
    *
    * @param newX   The set of observations if independent variables in 
    *               DATASET(NumericField) format.
    * @param model  A model that was previously returned from GetModel (above).
    *               Note that a model from a previous run will only be valid
    *               if the field numbers in X are the same as when the model
    *               was learned.
    * @return       An estimation of the corresponding Y value for each
    *               observation of newX.  Returned in DATASET(NumericField)
    *               format with field number (i.e. 'number') always = 1.
    */
  EXPORT DATASET(NumericField) Predict(DATASET(NumericField) newX, 
  									DATASET(Layout_Model) model) := FUNCTION
    mNewX := pbbConverted.NFToMatrix(newX);
    // Insert a column of ones for the Y intercept term
    mExtX := MatUtils.InsertCols(mNewX, 1, 1.0);
    // Multiply the new X values by Beta
    mNewY := PBblas.gemm(FALSE, FALSE, 1.0, mExtX, mBetas(model));
    NewY := pbbConverted.MatrixToNF(mNewY);
    return NewY;
  END;

  // The predicted values of Y using the given X
  mod := GetModel;
  SHARED DATASET(NumericField) modelY := Predict(X, mod);
  
  // Calculate the correlation coefficients (pearson) between
  // y and modelY (aka Yhat)
  SHARED Yhat := modelY;
  // Make record with Y * Yhat (i.e. YYhat)
  NumericField make_YYhat(NumericField l, NumericField r) := TRANSFORM
    SELF.value := l.value * r.value;
    SELF       := l;  // Arbitrary
  END;
  SHARED YYhat := JOIN(Y, Yhat, LEFT.wi=RIGHT.wi AND RIGHT.id=LEFT.id AND LEFT.number=RIGHT.number, 
                    make_YYhat(LEFT, RIGHT));
  // Calculate aggregates for Y, Yhat, and YYhat
  SHARED Y_stats := ML_Core.FieldAggregates(Y).simple;
  Yhat_stats := ML_Core.FieldAggregates(Yhat).simple;
  YYhat_stats := ML_Core.FieldAggregates(YYhat).simple;
  // Composite record of needed Y and Yhat stats
  CompRec0 := RECORD
    t_work_item   wi;
    t_FieldNumber number;
    t_FieldReal   e_Y;
    t_FieldReal   e_Yhat;
    t_FieldReal   sd_Y;
    t_FieldReal   sd_Yhat;
  END;
  //  Composite record of Y, Yhat, and YYhat stats
  CompRec := RECORD(CompRec0)
    t_FieldReal   e_YYhat;
  END;
  // Build the composite records
  CompRec0 make_comp0(Y_stats le, Yhat_stats ri) := TRANSFORM
    SELF.e_Y     := le.mean;
    SELF.e_Yhat  := ri.mean;
    SELF.sd_Y    := le.sd;
    SELF.sd_Yhat := ri.sd;
    SELF         := le;
  END;
  Ycomp0 := JOIN(Y_stats, Yhat_stats, LEFT.wi=RIGHT.wi AND LEFT.number=RIGHT.number, make_comp0(LEFT, RIGHT));

  CompRec make_comp(CompRec0 le, YYhat_stats ri) := TRANSFORM
    SELF.e_YYhat := ri.mean;
    SELF         := le;
  END;
  
  SHARED Ycomp  := JOIN(Ycomp0, YYhat_stats, LEFT.wi=RIGHT.wi and LEFT.number=RIGHT.number, make_comp(LEFT, RIGHT));

  // Correlation Coefficient Record
  SHARED CoCoRec := RECORD
    t_work_item   wi;
    t_FieldNumber number;
    Types.t_FieldReal   covariance;
    Types.t_FieldReal   pearson;
  END;

  // Form the Correlation Coefficient Records from the composite records
  CoCoRec MakeCoCo(Ycomp lr) := TRANSFORM
    SELF.covariance := (lr.e_YYhat - lr.e_Y*lr.e_Yhat);
    SELF.pearson := SELF.covariance/(lr.sd_Y*lr.sd_Yhat);
    SELF := lr;
  END;
  
  SHARED cor_coefs := PROJECT(Ycomp, MakeCoCo(LEFT));

  // The R Squared values for the parameters
  SHARED R2Rec := RECORD
    t_work_item wi;
    t_FieldReal   RSquared;
  END;

  R2Rec makeRSQ(CoCoRec coco) := TRANSFORM
    SELF.wi := coco.wi;
    SELF.RSquared := coco.pearson * coco.pearson;
  END;
  /**
    * RSquared
    *
    * Calculate the R Squared Metric used to assess the fit of the regression line to the 
    * training data.  Since the regression has chosen the best (i.e. least squared error) line
    * matching the data, this can be thought of as a measurement of the linearity of the
    * training data.  R Squared generally varies between 0 and 1, with 1 indicating an exact
    * linear fit, and 0 indicating that a linear fit will have no predictive power.  Negative
    * values are possible under certain conditions, and indicate that the mean(Y) will be more
    * predictive than any linear fit.  Moderate values of R squared (e.g. .5) may indicate
    * that the relationship of X -> Y is non-linear, or that the measurement error is high
    * relative to the linear correlation (e.g. many outliers).  In the former case, increasing
    * the dimensionality of X, such as by using polynomial variants of the features, may
    * yield a better fit.
    * 
    * Note that the result of this call is only meaningful during training phase (use case 1
    * above) as it is an analysis based on the training data which is not provided during a
    * prediction-only phase.
    *
    * @return  DATASET(R2Rec) with one record per work-item.
    *
    */
  EXPORT DATASET(R2Rec)  RSquared := PROJECT(cor_coefs, makeRSQ(LEFT));
  
  // Produce an Analysis of Variance report
  // Get the number of observations for each feature
  card := ML_Core.FieldAggregates(X).Cardinality;
  // Use that to calculate the number of features (k) for each work-item
  fieldCounts := TABLE(card, {wi,k:=COUNT(GROUP)}, wi);
  
  // Get basic stats for Y for each work-item
  tmpRec := RECORD
    RECORDOF(Y_stats);
    Types.t_fieldreal  RSquared;
    UNSIGNED           K; // Number of Independent variables
  END;
  // Combine the basic Y stats with R Squared
  Y_stats1 := JOIN(Y_stats, RSquared, LEFT.wi=RIGHT.wi,
          TRANSFORM(tmpRec,  SELF.RSquared := RIGHT.RSquared, SELF.K:=0,SELF := LEFT));
  // Include the number of dependent variables for each work-item (i.e. K)
  Y_stats2 := JOIN(Y_stats1, fieldCounts, LEFT.wi=RIGHT.wi, TRANSFORM(tmpRec, SELF.K:=RIGHT.K, SELF := LEFT), LOOKUP);
  AnovaRec := RECORD
    t_work_item           wi;
    Types.t_RecordID      Model_DF; // Degrees of Freedom
    Types.t_fieldreal      Model_SS; // Sum of Squares
    Types.t_fieldreal      Model_MS; // Mean Square
    Types.t_fieldreal      Model_F;  // F-value
    Types.t_RecordID      Error_DF; // Degrees of Freedom
    Types.t_fieldreal      Error_SS; // Sum of Squares
    Types.t_fieldreal      Error_MS; // Mean Square
    Types.t_RecordID      Total_DF; // Degrees of Freedom
    Types.t_fieldreal      Total_SS;  // Sum of Squares
  END;

  AnovaRec getResult(tmpRec le) :=TRANSFORM
    k   := le.K;
    SST := le.var*le.countval;
    SSM := SST*le.RSquared;
    SELF.wi       := le.wi;
    SELF.Total_SS := SST;
    SELF.Model_SS := SSM;
    SELF.Error_SS := SST - SSM;
    SELF.Model_DF := k;
    SELF.Error_DF := le.countval-k-1;
    SELF.Total_DF := le.countval-1;
    SELF.Model_MS := SSM/k;
    SELF.Error_MS := (SST - SSM)/(le.countval-k-1);
    SELF.Model_F  := (SSM/k)/((SST - SSM)/(le.countval-k-1));
  END;

  //http://www.stat.yale.edu/Courses/1997-98/101/anovareg.htm
  //Tested using the "Healthy Breakfast" dataset
  /**
    * ANOVA (Analysis of Variance) report
    * 
    * Analyzes the sources of variance for each field of the training
    * data.
    * This attribute is only meaningful during the training phase.
    *
    * Provides one record per field for each work-item.
    * Each record provides the following statistics:
    * - Total_SS -- Total Sum of Squares (SS) variance for the field
    * - Model_SS -- The SS variance represented within the model
    * - Error_SS -- The SS variance not reflected by the model
    *                (i.e. Total_SS - Error_SS)
    * - Model_MS -- The Mean Square (MS) variance represented within the
    *                model
    * - Error_MS -- The MS variance not reflected by the model
    * - Total_DF -- The total degrees of freedom within the dependent data
    * - Model_DF -- Degrees of freedom of the model
    * - Error_DF -- Degrees of freedom of the error component
    * - Model_F  -- The F-Test statistic Model_MS / Error_MS
    *
    * @return  DATASET(AnovaRec), one per field per work-item
    *
    */
  EXPORT Anova := PROJECT(Y_stats2, getResult(LEFT));

  // Compute the covariance matrix aka variance-covariance matrix of coefficients
  // Convert to matrix form for computation
  mXorig := PBblas.Converted.NFToMatrix(X); // Matrix of X values
  // We extend X by inserting a column of ones
  mX    := MatUtils.InsertCols(mXorig, 1, 1.0);
  
  // Calculate X transposed times X.  If the original X had shape N x M,
  // our extended XTX will have shape M+1 x M+1, with one row and column per
  // coefficient.
  mXTX := PBblas.gemm(TRUE, FALSE, 1.0, mX, mX);
  // Invert the matrix (solve for: XTX * X**-1 = X**T)
  mXT := PBblas.tran(1.0, mX);
  // Factor and two triangle solves gives us the solution for X**-1
  mXTX_fact := PBblas.potrf(triangle.Lower, mXTX);
  mXTX_S := PBblas.trsm(Side.Ax, Triangle.Lower, FALSE,
                       diagonal.NotUnitTri, 1.0, mXTX_fact, mXT);
  mXTXinv := PBblas.trsm(Side.Ax, triangle.Upper, TRUE,
                       diagonal.NotUnitTri, 1.0, mXTX_fact, mXTX_S);

  // Scale each cell by multiplying by the ANOVA Mean Square Error to arrive
  // at the covariance matrix of the coefficients.
  Layout_Cell scale_vc(Layout_Cell l, Anova r) := TRANSFORM
    SELF.v := l.v * r.Error_MS;
    SELF   := l;
  END;
  mCoef_covar := JOIN(mXTXinv, Anova, LEFT.wi_id = RIGHT.wi, scale_vc(LEFT, RIGHT), LOOKUP);
  
  /**
    * Coefficient Variance-Covariance Matrix
    *
    * This is the covariance matrix of the coefficients (i.e. Betas), not to be confused
    * with the covariance matrix for X.
    * If X is an N x M matrix (i.e. N observations of M features), then this will
    * be an M+1 x M+1 matrix, since there are M + 1 coefficients including the Y intercept.
    * Index 1 represents the Y intercept, Index 2 represents feature 1 .. Index M+1 represents
    * Feature M.  The diagonal entries designate the Variance of the coefficient across
    * observations, while the non-diagonal entries represents the covariance between two
    * coefficients.  For example, entry 2,3 represents the covariance between the coefficients
    * for feature 1 and feature 2.
    *
    * This supports the myriad interface and will produce separate matrices for each
    * work-item specified within X and Y.
    *
    * This is only meaningful during the training phase.
    *
    * @return  DATASET(NumericField) representing the covariance matrix of the coefficients
    *
    */
  // Convert matrix form to numeric field form
  EXPORT DATASET(NumericField) Coef_covar := PBBConverted.MatrixToNF(mCoef_covar);

  
  NumericField calc_sErr(NumericField lr) := TRANSFORM
    SELF.value := SQRT(lr.value);
    SELF.id    := 1; // Id is not used and set to one for consistency
    SELF       := lr;
  END;
  
  /**
    * Standard Error of the Regression Coefficients
    *
    * Describes the variability of the regression error for each coefficient.
    *
    * Only meaningful during the training phase.
    *
    * @return DATASET(NumericField), one record per Beta coefficient per work-item.
    *         The 'number' field is the coefficient number, with 1 being the
    *         Y intercept, 2 being the coefficient for the first feature, etc.
    *
    */
  // Standard error of the regression coefficients is just the square root of the
  // variance of each coefficient (i.e. the diagonal terms of the Coefficient
  // Covariance Matrix).
  EXPORT DATASET(NumericField) SE := PROJECT(Coef_covar(number=id), calc_sErr(LEFT));

  // Transformation to calculate the T Statistic
  NumericField tStat_transform(NumericField b, NumericField s) := TRANSFORM
    SELF.value := b.value / s.value;
    SELF := b;
  END;
  /**
    * T Statistic
    *
    * The T statistic identifies the significance of the value of each regression
    * coefficient.  Its calculation is simply the value of the coefficient divided
    * by the Standard Error of the coefficient.
    *
    * Only meaningful during the training phase.
    * 
    * @return DATSET(NumericField), one record per Beta coefficient per work-item.
    *         The 'number' field is the coefficient number, with 1 being the
    *         Y intercept, 2 being the coefficient for the first feature, etc. 
    */  
  EXPORT DATASET(NumericField) tStat := JOIN(Betas(), SE, LEFT.wi = RIGHT.wi AND
                                    LEFT.id = RIGHT.id AND 
                                    LEFT.number = RIGHT.number, 
                                    tStat_transform(LEFT, RIGHT));

  // Calculate Adjusted R Squared
  R2Rec adjustR2(Rsquared l, Anova r) := TRANSFORM
    SELF.RSquared := 1 - ( 1 - l.RSquared) * (r.Total_DF / r.Error_DF);
    SELF          := l;
  END;

  /**
    * Adjusted R2
    *
    * Calculate Adjusted R Squared which is a scaled version of R Squared
    * that does not arbitrarily increase with the number of features.
    * Adjusted R2, rather than R2 should always be used when trying to determine the
    * best set of features to include in a model.  When adding features, R2 will
    * always increase, whether or not it improves the predictive power of the model.
    * Adjusted R2, however, will only increase with the predictive power of the model.
    * 
    * @return  DATASET(R2Rec), one record per work-item
    *
    */
  EXPORT DATASET(R2Rec) AdjRSquared := JOIN(Rsquared, Anova, 
                          LEFT.wi=RIGHT.wi,
                          adjustR2(LEFT, RIGHT), LOOKUP);
 

  
  // Record format for AIC results
  AICRec := RECORD
    t_work_item wi;
    Types.t_FieldReal AIC;
  END;

  /**
    * Akaike Information Criterion (AIC)
    *
    * Information theory based criterion for assessing Goodness of Fit (GOF).
    * Lower values mean better fit.
    *
    * @return DATASET(AICRec), one record per work-item
    *
    */
  EXPORT DATASET(AICRec) AIC := PROJECT(Anova, TRANSFORM(AICRec, 
            n := LEFT.Total_DF + 1;
            p := LEFT.Model_DF + 1;
            SELF.AIC := n * LN(LEFT.Error_SS / n) + 2 * p; 
            SELF := LEFT));




















  // Density vector
  EXPORT RangeVec := RECORD
    t_Count RangeNumber;
    t_FieldReal RangeLow; // Values > RangeLow
    t_FieldReal RangeHigh; // Values <= RangeHigh
    t_FieldReal P;
  END;
  
  // The 'double' factorial is defined for ODD n and is the product of all the odd numbers up to and including that number
  // We are extending the meaning to even numbers to mean the product of the even numbers up to and including that number
  // Thus DoubleFac(8) = 8*6*4*2
  // We also defend against i < 2 (returning 1.0)
  EXPORT REAL8 DoubleFac(INTEGER2 i) := BEGINC++
    if ( i < 2 )
      return 1.0;
    double accum = (double)i;
	  for ( int j = i-2; j > 1; j -= 2 )
		accum *= (double)j;
	return accum;
  ENDC++;
  EXPORT Pi := 3.1415926535897932384626433;
  
  // We temporarily include code here for needed statistical distributions.
  // This should be removed once we have a productized Distribution module.
  EXPORT DefaultDist := MODULE,VIRTUAL
    EXPORT RangeWidth := 1.0; // Only really works for discrete - the width of each range
    EXPORT t_FieldReal Density(t_FieldReal RH) := 0.0; // Density function at stated point
    // Generating functions are responsible for making these in ascending order
    EXPORT DensityV() := DATASET([],RangeVec); // Probability of between >PreviosRangigh & <= RangeHigh
    // Default CumulativeV works by simple integration of the DensityVec
    EXPORT CumulativeV() := FUNCTION
      d := DensityV();
      RangeVec Accum(RangeVec le,RangeVec ri) := TRANSFORM
        SELF.p := le.p+ri.p*RangeWidth;
        SELF := ri;
      END;
      RETURN ITERATE(d,Accum(LEFT,RIGHT)); // Global iterates are horrible - but this should be tiny
    END;
    // Default Cumulative works from the Cumulative Vector
    EXPORT t_FieldReal Cumulative(t_FieldReal RH) :=FUNCTION // Cumulative probability at stated point
      cv := CumulativeV();
      // If the range high value is at an intermediate point of a range then interpolate the result\
      // Interpolation done as follows :
      // cumulative(RH) = cumulative(v.RangeHigh) - prob(RH <= x < v.Rangehigh)
      // prob(RH <= x < v.Rangehigh) =  Density((RH + Rangehigh)/2) * (RH - Rangehigh) [Rectangle Rule for integration]
      InterC(RangeVec v) := IF ( RH=v.RangeHigh, v.P, v.P - Density((v.RangeHigh+RH)/2)*(v.RangeHigh - RH));
      RETURN MAP( RH >= MAX(cv,RangeHigh) => 1.0,
                  RH <= MIN(cv,RangeLow) => 0.0,
                  InterC(cv(RH>RangeLow,RH<=RangeHigh)[1]) );
    END;
    // Default NTile works from the Cumulative Vector
    EXPORT t_FieldReal NTile(t_FieldReal Pc) :=FUNCTION // Value of the Pc percentile
      cp := Pc / 100.0; // Convert from percentile to cumulative probability
      cv := CumulativeV();
      // If the range high value is at an intermediate point of a range then interpolate the result
      InterP(RangeVec v) := IF ( cp=v.P, v.RangeHigh, v.RangeHigh+(cp-v.p)/Density((v.RangeHigh+v.RangeLow)/2) );
      RETURN MAP( cp >= MAX(cv,P) => MAX(cv,RangeHigh),
                  cp <= 0.0 => MIN(cv,RangeLow),
                  InterP(cv(P>=cp)[1]) );
    END;
    EXPORT InvDensity(t_FieldReal delta) := 0.0; //Only sensible for monotonic distributions
    EXPORT Discrete := FALSE;
  END;
  
  // Student T distribution
  // This distribution is entirely symmetric about the mean - so we will model the >= 0 portion
  // Warning - v=1 tops out around the 99.5th percentile, 2+ is fine

  EXPORT StudentT(t_Discrete v,t_Count NRanges = 10000) := MODULE(DefaultDist)
    // Used for storing a vector of probabilities (usually cumulative)
    EXPORT Layout := RECORD
      t_Count RangeNumber;
      t_FieldReal RangeLow; // Values > RangeLow
      t_FieldReal RangeHigh; // Values <= RangeHigh
      t_FieldReal P;
    END;
    SHARED Multiplier := IF ( v & 1 = 0, DoubleFac(v-1)/(2*SQRT(v)*DoubleFac(v-2))
                             , DoubleFac(v-1)/(Pi*SQRT(v)*DoubleFac(v-2)));
    // Compute the value of t for which a given density is obtained                   
    SHARED LowDensity := 0.00001; // Go down as far as a density of 1x10-5
    EXPORT InvDensity(t_FieldReal delta) := SQRT(v*(EXP(LN(delta/Multiplier)*-2.0/(v+1))-1));
    // We are defining a high value as the value at which the density is 'too low to care'
    SHARED high := InvDensity(LowDensity);
    SHARED Low := 0;
    EXPORT RangeWidth := (high-low)/NRanges;
    // Generating functions are responsible for making these in ascending order
    EXPORT t_FieldReal Density(t_FieldReal RH) := Multiplier * POWER( 1+RH*RH/v,-0.5*(v+1) );
    EXPORT DATASET(RangeVec) DensityV() := FUNCTION
      dummy := DATASET([{0,0,0,0}], RangeVec);
      RangeVec make_density(RangeVec d, UNSIGNED c) := TRANSFORM
        SELF.RangeNumber := c;
        SELF.RangeLow := Low + (c-1) * RangeWidth;
        SELF.RangeHigh := SELF.RangeLow + RangeWidth;
        SELF.P := Density((SELF.RangeLow + SELF.RangeHigh) / 2);
      END;
      vec := NORMALIZE(dummy, Nranges, make_density(LEFT, COUNTER));
      RETURN vec;
    END;
    EXPORT CumulativeV() := FUNCTION
      d := DensityV();
      // The general integration really doesn't work for v=1 and v=2 - fortunately there are 'nice' closed forms for the CDF for those values of v
      Layout Accum(Layout le,Layout ri) := TRANSFORM
        SELF.p := MAP( v = 1 => 0.5+ATAN(ri.RangeHigh)/PI, // Special case CDF for v = 1
                       v = 2 => (1+ri.RangeHigh/SQRT(2+POWER(ri.RangeHigh,2)))/2, // Special case of CDF for v=2
                       IF(le.p=0,0.5,le.p)+ ri.p*RangeWidth );
        SELF := ri;
      END;
      
      RETURN ITERATE(d,Accum(LEFT,RIGHT)); // Global iterates are horrible - but this should be tiny
    END;
    // Cumulative works from the Cumulative Vector
    EXPORT t_FieldReal Cumulative(t_FieldReal RH) :=FUNCTION // Cumulative probability at stated point
      cv := CumulativeV();
      // If the range high value is at an intermediate point of a range then interpolate the result\
      // Interpolation done as follows :
      // cumulative(RH) = cumulative(v.RangeHigh) - prob(RH <= x < v.Rangehigh)
      // prob(RH <= x < v.Rangehigh) =  Density((RH + Rangehigh)/2) * (RH - Rangehigh) [Rectangle Rule for integration]
      InterC(Layout v) := IF ( RH=v.RangeHigh, v.P, v.P - Density((v.RangeHigh+RH)/2)*(v.RangeHigh - RH));
      RETURN MAP( RH >= MAX(cv,RangeHigh) => 1.0,
                RH <= MIN(cv,RangeLow) => 0.0,
                InterC(cv(RH>RangeLow,RH<=RangeHigh)[1]) );
    END;
    // Default NTile works from the Cumulative Vector
    EXPORT t_FieldReal NTile(t_FieldReal Pc) :=FUNCTION // Value of the Pc percentile
      cp := Pc / 100.0; // Convert from percentile to cumulative probability
      cv := CumulativeV();
      // If the range high value is at an intermediate point of a range then interpolate the result
      InterP(Layout v) := IF ( cp=v.P, v.RangeHigh, v.RangeHigh+(cp-v.p)/Density((v.RangeHigh+v.RangeLow)/2) );
      RETURN MAP( cp >= MAX(cv,P) => MAX(cv,RangeHigh),
                  cp <= 0.0 => MIN(cv,RangeLow),
                  InterP(cv(P>=cp)[1]) );
    END;
  END;
  EXPORT FDist(t_Discrete d1, t_Discrete d2, t_Count NRanges = 10000) := MODULE(DefaultDist)
    SHARED Multiplier := (1 / Utils.Beta(d1/2, d2/2)) * POWER(d1/d2, d1/2);
    SHARED high := 15;
    SHARED Low := 0;
    EXPORT RangeWidth := (high - low)/NRanges;
    EXPORT t_FieldReal Density(t_FieldReal RH) := Multiplier * POWER(RH, d1/2 - 1) / POWER(1 + d1 * RH/d2, (d1 + d2)/2);
    EXPORT DensityV() := PROJECT(DVec(NRanges,low,RangeWidth),
                         TRANSFORM(Layout,
                           SELF.P := Density((LEFT.RangeLow+LEFT.RangeHigh)/2),
                           SELF := LEFT));  
  END;
  EXPORT dist := StudentT(Anova[1].Error_DF, 100000);
  
  NumericField pVal_transform(NumericField b) := TRANSFORM 
    SELF.value := 2 * ( 1 - dist.Cumulative(ABS(b.value))); 
    SELF := b;
  END;
  
  EXPORT pVal := PROJECT(tStat, pVal_transform(LEFT));
    
  confintRec := RECORD
    Types.t_RecordID id;
    Types.t_FieldNumber number;
    Types.t_Fieldreal LowerInt;
    Types.t_Fieldreal UpperInt;
  END;
  
  confintRec confint_transform(NumericField b, NumericField s, REAL Margin) := TRANSFORM
    SELF.UpperInt := b.value + Margin * s.value;
    SELF.LowerInt := b.value - Margin * s.value;
    SELF := b;
  END;
                                
  EXPORT ConfInt(Types.t_fieldReal level) := FUNCTION
    newlevel := 100 - (100 - level)/2;
    Margin := dist.NTile(newlevel);
    RETURN JOIN(Betas(), SE, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, 
                     confint_transform(LEFT,RIGHT,Margin));
  END;                                
  FTestRec := RECORD
    Types.t_FieldReal Model_F;
    Types.t_FIeldReal pValue;
  END;

  EXPORT DATASET(FTestRec) FTest := PROJECT(Anova, TRANSFORM(FTestRec, SELF.Model_F := LEFT.Model_F;
                  dist := ML.Distribution.FDist(LEFT.Model_DF, LEFT.Error_DF, 100000);
                  SELF.pValue := 1 - dist.cumulative(LEFT.Model_F)));
  **/
END;
