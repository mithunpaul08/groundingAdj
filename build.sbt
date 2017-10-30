

name := "gradability-wm"

version := "1.0"

scalaVersion := "2.11.8"



organization := "org.clulab"



scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation")

//
// publishing settings
//

//// these are the steps to be performed during release
//releaseProcess := Seq[ReleaseStep](
//  checkSnapshotDependencies,
//  inquireVersions,
//  runClean,
//  runTest,
//  setReleaseVersion,
//  commitReleaseVersion,
//  tagRelease,
//  ReleaseStep(action = Command.process("publishSigned", _)),
//  setNextVersion,
//  commitNextVersion,
//  ReleaseStep(action = Command.process("sonatypeReleaseAll", _)),
//  pushChanges
//)

// publish to a maven repo
publishMavenStyle := true

// the standard maven repository
publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

// let’s remove any repositories for optional dependencies in our artifact
pomIncludeRepository := { _ => false }

// mandatory stuff to add to the pom for publishing
pomExtra := (
  <url>https://github.com/clulab/reach</url>
    <licenses>
      <license>
        <name>Apache License, Version 2.0</name>
        <url>http://www.apache.org/licenses/LICENSE-2.0.html</url>
        <distribution>repo</distribution>
      </license>
    </licenses>
    <scm>
      <url>https://github.com/clulab/reach</url>
      <connection>https://github.com/clulab/reach</connection>
    </scm>
    <developers>
      <developer>
        <id>mithun.paul</id>
        <name>Mithun Paul</name>
        <email>mithunpaul@email.arizona.edu</email>
      </developer>
    </developers>)

//
// end publishing settings
//
resolvers += "lightshed-maven" at "http://dl.bintray.com/content/lightshed/maven"

libraryDependencies ++= Seq(

  "org.clulab" %% "processors-main" % "6.0.0",
  "org.clulab" %% "processors-models" % "6.0.0",
  "org.clulab" %% "processors-corenlp" % "6.0.0",
  "com.typesafe" % "config" % "1.2.1",
  "commons-io" % "commons-io" % "2.4",
  "com.github.myedibleenso" %% "processors-agiga" % "0.0.2",

  //// logging
  "com.typesafe.scala-logging" %% "scala-logging" % "3.5.0",
  "javax.mail" % "mail" % "1.4",
  "ch.qos.logback" % "logback-classic" % "1.1.7"


)
