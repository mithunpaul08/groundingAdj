package initializer

import org.clulab.embeddings.word2vec.Word2Vec

object initializer extends App {


  val data = io.Source.fromFile("src/main/resources/turk_fullText.txt").getLines().toArray
  val split_data = data.tail.map { e =>
    val fields = e.split("\t")
    val adj = fields(0)
    val intercept = fields(3).toFloat
    (adj, intercept)
  }
  val sorted_split_data = split_data.sortBy(_._2)

  val objW2v = Word2Vec;
  val w2v = new Word2Vec("//data/nlp/corpora/word2vec/gigaword/vectors.txt")

  val firstAdj= sorted_split_data(0)._1

//  for ((adj, intercept) <- sorted_split_data) {
//    val adj_sanitized = objW2v.sanitizeWord(adj);
//    print(adj_sanitized)
//    val sim=w2v.similarity(firstAdj,adj_sanitized);
//
//  }

  val w2vSim=sorted_split_data.map { d=>

    val adj= d._1;
    val intercept=d._2
    val adj_sanitized = objW2v.sanitizeWord(adj);
    print(adj_sanitized)
    val sim=w2v.similarity(firstAdj,adj_sanitized);
    (sim)

  }

  println(w2vSim.mkString("\n"))
}