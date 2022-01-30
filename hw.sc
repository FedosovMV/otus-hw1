import breeze.linalg._
import breeze.numerics._
import breeze.stats.regression.leastSquares
import breeze.stats.meanAndVariance
import breeze.stats.mean
import breeze.stats.variance
import breeze.plot._
import breeze.stats.covmat


//Часть 1. Скачайте данные
val dataFile = io.Source.fromFile("E:/otus/data/archive/Dataset_hw1_bkp.csv")

val dataArray = dataFile.getLines.drop(1)
  .map(_.split(",").filter(_ != "").map(_.trim))
  .map { line => line.map { elem =>
    elem match {
      case "" => 0.0
      case x => x.toInt
    }
  }
  }.toArray

val dataMatrix = DenseMatrix(dataArray: _*)

println("Dim: " + dataMatrix.rows + " " + dataMatrix.cols)




//2. Исследование данных
def corr(a: DenseVector[Double], b: DenseVector[Double]): Double = {
  if (a.length != b.length)
    sys.error("error")

  val n = a.length

  val ameanavar = meanAndVariance(a)
  val amean = ameanavar.mean
  val avar = ameanavar.variance
  val bmeanbvar = meanAndVariance(b)
  val bmean = bmeanbvar.mean
  val bvar = bmeanbvar.variance
  val astddev = math.sqrt(avar)
  val bstddev = math.sqrt(bvar)

  1.0 / (n - 1.0) * sum( ((a - amean) / astddev) * ((b - bmean) / bstddev) )
}



//val  qwe  = corr(y_train,x_train(::,1))

//val  covariance  = covmat(x_train)
//println(covariance.toString(27,Int.MaxValue))

val dataMatrix_upd = dataMatrix.delete(Seq(11,14,17,21), Axis._1)


//Часть 3. Моделирование

val max_index_train = Math.round(dataMatrix_upd.rows*8/10)
val start_index_test = max_index_train +1


val y_train = dataMatrix_upd(0 to max_index_train , dataMatrix_upd.cols -1).toDenseVector
val y_test = dataMatrix_upd(start_index_test to dataMatrix_upd.rows-1 , dataMatrix_upd.cols -1).toDenseVector

val x_train = dataMatrix_upd(0 to max_index_train, 0 to dataMatrix_upd.cols -2)
val x_test = dataMatrix_upd(start_index_test to dataMatrix_upd.rows-1, 0 to dataMatrix_upd.cols -2)

val result = leastSquares(x_train, y_train)
val u= sum(pow(y_train - result(x_train), 2))
val y_train_meanavar = meanAndVariance(y_train)
val y_train_mean = y_train_meanavar.mean
val v = sum(pow(y_train - y_train_mean, 2))
val r2_adj = 1- (1-u/v)*(max_index_train-1)/(max_index_train-dataMatrix_upd.cols-1)

//почему то result.rSquared дает некорректный результат



val sse_test  = sum(pow(y_test - result(x_test), 2))
val mse_test   = sum(pow(y_test - result(x_test), 2))/(dataMatrix_upd.rows-1-start_index_test)
val rmse_test   = sqrt(sum(pow(y_test - result(x_test), 2))/(dataMatrix_upd.rows-1-start_index_test))



println(sse_test)
println(mse_test)
println(rmse_test)

//Часть 4. Реализация линейной регрессии руками
//Y=bX+c


val b = pinv(x_train.t * x_train) * x_train.t * y_train

//metrics by train
val sse_by_hand_train  = sum(pow(y_train - x_train*b, 2))
val mse_by_hand_train  = sum(pow(y_train - x_train*b, 2))/(max_index_train)
val rmse_by_hand_train  = sqrt(sum(pow(y_train - x_train*b, 2))/(max_index_train))

println(sse_by_hand_train)
println(mse_by_hand_train)
println(rmse_by_hand_train)

//metrics by test
val sse_by_hand_test  = sum(pow(y_test - x_test*b, 2))
val mse_by_hand_test   = sum(pow(y_test - x_test*b, 2))/(dataMatrix_upd.rows-1-start_index_test)
val rmse_by_hand_test   = sqrt(sum(pow(y_test - x_test*b, 2))/(dataMatrix_upd.rows-1-start_index_test))

println(sse_by_hand_test)
println(mse_by_hand_test)
println(rmse_by_hand_test)