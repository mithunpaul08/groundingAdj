package initializer

import java.io.{BufferedWriter, File, FileWriter}

object utils {

  def writeToFile(stringToWrite: String, outputFilename: String, outputDirectoryPath: String): Unit = {
    val outFile = new File(outputDirectoryPath, outputFilename)
    val bw = new BufferedWriter(new FileWriter(outFile))
    bw.write(stringToWrite)
    bw.close()

  }
}
