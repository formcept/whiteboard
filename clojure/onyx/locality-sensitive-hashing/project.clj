(defproject org.formcept.onyx/lsh-onyx "1.0.0"
  :description "Contains implementation of locality-sensitive hashing on Onyx"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 ^{:voom {:repo "git@github.com:onyx-platform/onyx.git" :branch "0.10.x"}}
                 [org.onyxplatform/onyx "0.10.0-beta17"]
                 [org.formcept.onyx/onyx-utils "1.0.0"]]
  :plugins [[lein-update-dependency "0.1.2"]]
  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :main org.formcept.onyx.lsh-onyx.core
  :profiles {:ship {:aot :all}})

