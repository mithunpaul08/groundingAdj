package adjVectorCompare

import java.io.{BufferedWriter, File, FileWriter}
import org.clulab.embeddings.word2vec.Word2Vec
import utils._
import w2v._

object initializer extends App {
  val w2v = "//data/nlp/corpora/word2vec/gigaword/vectors.txt";
  val glove = "//data/nlp/corpora/glove/6B/glove.6B.200d.txt"
  val marneffe1 = "/data/nlp/corpora/demarneffe/cbow_vectors_syn_ant_sameord_difford.txt"

 val outputFileW2v="comparew2vsim.txt"
  val outputFileGlove="turkGlove.txt"
  val outputFileMarneffe="turkMarneffe.txt"


  //for each adj in the turk task adjectives get and sort by its intercept
  val data = io.Source.fromFile("src/main/resources/turk_fullText.txt").getLines().toArray
  val split_data = data.tail.map { e =>
    val fields = e.split("\t")
    val adj = fields(0)
    val intercept = fields(3).toFloat
    (adj, intercept)
  }
  val sorted_split_data = split_data.sortBy(_._2)
  writeToFile(sorted_split_data.mkString("\n"), "Turksim.txt", "src/main/outputs/")
  //pick the adjective with the smallest intercept
  val firstAdj = sorted_split_data(0)._1
  println(sorted_split_data.mkString("\n"))
  calculateSimilarity(marneffe1,firstAdj,outputFileMarneffe,sorted_split_data)
}