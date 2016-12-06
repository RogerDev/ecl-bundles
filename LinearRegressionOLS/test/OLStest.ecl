IMPORT LinearRegressionOLS as LROLS;
IMPORT MLCore;
IMPORT MLCore.Types as mlTypes;
IMPORT PBBlas.test.MakeTestMatrix as tm;
IMPORT PBBlas.Types as pbbTypes;
IMPORT PBBlas.Converted as pbbConverted;
Layout_Cell := pbbTypes.Layout_Cell;
NumericField := mlTypes.NumericField;
two31 := POWER(2, 31);
REAL Noise := FUNCTION
  out := ((RANDOM()-two31)%1000)/10000;
  return out;
END;
Layout_Cell makeY(Layout_Cell X, REAL A, REAL B) := TRANSFORM
  SELF.x := X.x;
  SELF.y := 1;
  SELF.wi_id := X.wi_id;
  noiseY := Noise;
  noisyA := A + Noise;
  noisyB := B + Noise;
  SELF.v := noisyA* X.v + noisyB + noiseY;
END;
test_rslt := RECORD
  STRING32 TestName;
  SET OF REAL X;
  REAL Y;
  REAL projY;
  REAL diff;
  REAL pctErr;
END;

// TEST 1 -- Simple Linear Regression
A1 := 3.123;
B1 := -1.222;
N1 := 10000;
M1 := 1;
mX1 := tm.Random(N1, M1, 1.0, 1);
mY1 := PROJECT(mX1, makeY(LEFT, A1, B1));
X1 := pbbConverted.MatrixToNF(mX1);
Y1 := pbbConverted.MatrixToNF(mY1);
lr1 := LROLS.OLS(X1, Y1);
model1 := lr1.GetModel;
newmX1 := tm.Random(50, M1, 1.0, 1);
newX1 := pbbConverted.MatrixToNF(newmX1);
predY1 := lr1.Predict(newX1, model1);
test_rslt formatRslt1(STRING32 tn, NumericField l, NumericField r) := TRANSFORM
  SELF.TestName := tn;
  SELF.X := [l.value];
  SELF.Y := r.value;
  SELF.projY := A1 * l.value + B1;
  SELF.diff := SELF.Y - SELF.projY;
  SELF.pctErr := SELF.diff / SELF.projY;
END;
rslt1 := JOIN(newX1, predY1, LEFT.id=RIGHT.id, formatRslt1('TEST 1 -- Simple Regression', LEFT, RIGHT));

// TEST2 -- Multiple Regression
compX2 := RECORD
  REAL wi;
  REAL id;
  REAL X1;
  REAL X2;
  REAL X3;
END;
compX2 makeComposite2(Layout_Cell l, DATASET(Layout_Cell) r) := TRANSFORM
  SELF.wi := l.wi_id;
  SELF.id := l.x;
  SELF.X1 := r(y=1)[1].v;
  SELF.X2 := r(y=2)[1].v;
  SELF.X3 := r(y=3)[1].v;
END;
A21 := -1.8;
A22 := -4.333;
A23 := 11.13;
B2 := -3.333;
N2 := 10000;
M2 := 3;
mX2 := tm.Random(N2, M2, 1.0, 2);
sX2 := SORT(mX2, wi_id, x);
gX2 := GROUP(sX2, wi_id, x);
cX2 := ROLLUP(gX2,  GROUP, makeComposite2(LEFT, ROWS(LEFT)));
Layout_Cell makeY2(compX2 X) := TRANSFORM
  SELF.x := X.id;
  SELF.y := 1;
  SELF.wi_id := X.wi;
  noiseY := Noise;
  noisyA1 := A21 + Noise;
  noisyA2 := A22 + Noise;
  noisyA3 := A23 + Noise;
  noisyB := B2 + Noise;
  SELF.v := noisyA1* X.X1 + noisyA2 * X.X2 + noisyA3 * X.X3 + noisyB + noiseY;
END;
mY2 := PROJECT(cX2, makeY2(LEFT));
X2 := pbbConverted.MatrixToNF(mX2);
Y2 := pbbConverted.MatrixToNF(my2);
lr2 := LROLS.OLS(X2, Y2);
model2 := lr2.GetModel;
newmX2 := tm.Random(50, M2, 1.0, 2);
newX2 := pbbConverted.MatrixToNF(newmX2);
predY2 := lr2.Predict(newX2, model2);

sNewX2 := SORT(newmX2, wi_id, x);
gNewX2 := GROUP(sNewX2, wi_id, x);
cNewX2 := ROLLUP(gNewX2,  GROUP, makeComposite2(LEFT, ROWS(LEFT)));
test_rslt formatRslt2(STRING32 tn, compX2 l, NumericField r) := TRANSFORM
  SELF.TestName := tn;
  SELF.X := [l.X1, l.X2, l.X3];
  SELF.Y := r.value;
  SELF.projY := B2 + A21 * l.X1 + A22 * l.X2 + A23 * l.X3;
  SELF.diff := SELF.Y - SELF.projY;
  SELF.pctErr := ABS(SELF.diff / SELF.projY);
END;
DATASET(test_rslt) rslt2 := JOIN(cNewX2, predY2, LEFT.id=RIGHT.id, formatRslt2('TEST 2 -- Multiple Regression', LEFT, RIGHT));

// TEST 3 -- Myriad -- Test 1 and Test 2 simultaneously
lr3 := LROLS.OLS(X1 + X2, Y2 + Y1);
model3 := lr3.GetModel;

predY3 := lr3.Predict(newX2 + newX1, model3);

predY31 := predY3(wi=1);
predY32 := predY3(wi=2);

rslt31 := JOIN(newX1, predY31, LEFT.id=RIGHT.id, formatRslt1('TEST 31 -- Myriad(1)', LEFT, RIGHT));
rslt32 := JOIN(cNewX2, predY32, LEFT.id=RIGHT.id, formatRslt2('TEST 32 -- Myriad(2)', LEFT, RIGHT));

rslt := rslt1 + rslt2 + rslt31 + rslt32;
test_summary := RECORD
  testname := rslt.testname;
  REAL     maxErr := MAX(GROUP, rslt.pctErr);
  REAL     avgErr := AVE(GROUP, rslt.pctErr);
END;
summary := TABLE(rslt, test_summary, testname);
EXPORT OLStest := summary;

