package adjVectorCompare

import java.io.{BufferedWriter, File, FileWriter}
import utils._;
import org.clulab.embeddings.word2vec.Word2Vec


object w2v {

  def calculateSimilarity(w2vFilePath: String,  firstAdj:String, outputFileName:String,sorted_split_data:Array[(String, Float)]): Unit = {
    val objW2v = Word2Vec;
    val w2v = new Word2Vec(w2vFilePath)

    //for each of the adj in the above list, get its embeddings value and compare with firstadj using cosine similarity
    val w2vSim = sorted_split_data.map { d =>

      val adj = d._1;
      val intercept = d._2
      val adj_sanitized = objW2v.sanitizeWord(adj);
      val firstAdj_sanitized = objW2v.sanitizeWord(firstAdj);
      print(adj_sanitized)
      val sim = w2v.similarity(firstAdj_sanitized, adj_sanitized);
      (adj_sanitized, sim)
    }
    val sorted_w2vSim = w2vSim.sortBy(-_._2)
    println(sorted_w2vSim.mkString("\n"))

    writeToFile(sorted_w2vSim.mkString("\n"), outputFileName, "src/main/outputs/")

  }
}
