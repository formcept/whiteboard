(defproject spark "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [yieldbot/flambo "0.8.3-SNAPSHOT"
                  :exclusions [[com.esotericsoftware/kryo-shaded]
                               [com.esotericsoftware/kryo]
                               [com.twitter/chill_2.11]]]]
  :profiles {:provided
             {:dependencies
              [[org.apache.spark/spark-core_2.11 "2.4.4"]
               [org.apache.spark/spark-sql_2.11 "2.4.4"]]}}
  :aot [spark.core])
