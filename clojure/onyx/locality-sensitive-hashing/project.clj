(defproject org.formcept.onyx/lsh-onyx "1.0.0"
  :description "Contains implementation of locality-sensitive hashing on Onyx"
  :url "https://github.com/formcept/wisdom/tree/dev/ml-onyx/locality-sensitive-hashing"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 ^{:voom {:repo "git@github.com:onyx-platform/onyx.git" :branch "0.10.x"}}
                 [org.onyxplatform/onyx "0.10.0-beta17"]
                 [org.formcept.onyx/onyx-utils "1.0.0"]]
  :plugins [[lein-update-dependency "0.1.2"]]
  :main org.formcept.onyx.lsh-onyx.core
  :profiles {:ship {:aot :all}})

