package kaggle.expedia

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.{SparkContext,	SparkConf}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

object Expedia {
  def main(args: Array[String]): Unit = {
    // val	conf	=	new	SparkConf().setAppName("Kaggle").setMaster("local[4]")
    val	conf	=	new	SparkConf().setAppName("Kaggle")
    val	sc	=	new	SparkContext(conf)
    val	sqlContext	=	new	SQLContext(sc)

    // print train's fields
    val	data	=	sc.textFile("train.csv")
    val	head	=	data.first()
    var	i	=	-1
    val	field	=	head.split(",").map{	x	=>
      i	+=	1
      (x,	i)
    }.foreach(println)

    // print test's fields
    val	data2	=	sc.textFile("test.csv")
    val	head2	=	data2.first()
    var	i2	=	-1
    val	field2	=	head2.split(",").map{	x	=>
      i2	+=	1
      (x,	i2)
    }.foreach(println)

    val	is_book	=	data.filter(_	!=	head).map(_.split(",")).filter(_(18)	!=	"0")
    val	train	=	is_book.map{	x	=>
      LabeledPoint(x(23).toDouble,	Vectors.dense(
        x(1).toInt,
        x(2).toInt,
        x(3).toInt,
        x(4).toInt,
        x(5).toInt,
        x(8).toInt,
        x(9).toInt,
        x(10).toInt,
        x(13).toInt,
        x(14).toInt,
        x(15).toInt,
        x(16).toInt,
        x(17).toInt,
        x(20).toInt,
        x(21).toInt,
        x(22).toInt
      ))
    }
    val	test	=	data2.filter(_	!=	head2).map{x	=>
      val	part	=	x.split(",")
      (part(0),	Vectors.dense(
        part(2).toInt,
        part(3).toInt,
        part(4).toInt,
        part(5).toInt,
        part(6).toInt,
        part(9).toInt,
        part(10).toInt,
        part(11).toInt,
        part(14).toInt,
        part(15).toInt,
        part(16).toInt,
        part(17).toInt,
        part(18).toInt,
        part(19).toInt,
        part(20).toInt,
        part(21).toInt
      ))
    }

//    val model = NaiveBayes.train(train, lambda = 1.0, modelType = "multinomial")

    val	numClasses	=	100
    val	categoricalFeaturesInfo	=	Map[Int,	Int]()
    val	impurity	=	"gini"
    val	maxDepth	=	10
    //	maximum	number	of	bins	used	for	spliHng	features
    val	maxBins	=	32
    val	model	=	DecisionTree.trainClassifier(train,	numClasses,
      categoricalFeaturesInfo,
      impurity,	maxDepth,	maxBins)

    val	pred_label	=	test.map{case	(id,	feat)	=>
      //	element	in	test	is	form	of	LabelPoint(,)
      val	pred	=	model.predict(feat)
      //	use	feature	in	test	to	predict	label
      (id,	pred.toInt)
    }
//    pred_label.foreach(println)

    val df = sqlContext.createDataFrame(pred_label.coalesce(1,true)).toDF("id", "hotel_cluster")
    df.write
      .format("com.databricks.spark.csv")
      .option("header",	"true")
      .save("result")
  }
}
