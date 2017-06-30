;;
;; Copyright (c) 2011-onwards, FORMCEPT [http://www.formcept.com]
;; All rights reserved.
;;
;; NOTICE:  All information contained herein is, and remains
;; the property of FORMCEPT and its suppliers, if any.
;; The intellectual and technical concepts contained
;; herein are proprietary to FORMCEPT and its suppliers and
;; may be covered by U.S. and Foreign Patents, patents in process,
;; and are protected by trade secret or copyright law. Dissemination
;; of this information or reproduction of this material
;; is strictly forbidden unless prior written permission is obtained
;; from FORMCEPT.
;;

(ns org.formcept.onyx.onyx-utils.string-utils.core
  (:require [æsahættr :as hasher]))

;; String Similarity

(defn levenstein-distance
  [s1 s2]
  (let [similarity (atom 0)
        edit-map (atom {})]
    (do 
      (doseq [i (range (count s1))]
        (let [edit-dist (atom (if (= i 0) 0 i))]
          (doseq [j (range (count s2))]
            (let [e (if (= (.charAt s1 i) (.charAt s2 j)) 0 1)]
              (do
                (reset! edit-dist 
                  (min 
                    (if (= j 0) (+ i 2) (inc @edit-dist)) 
                      (if (= i 0) (+ j 2) (inc (@edit-map (str (dec i) "," j))) )
                        (+ (if (= i 0) j (if (= j 0) i (@edit-map (str (dec i) "," (dec j))))) 
                           (if (and (= i 0) (= j 0)) 0 e))))
                (swap! edit-map assoc (str i "," j) @edit-dist)
                (when (and (= i (dec (count s1))) (= j (dec (count s2))))
                  (reset! edit-map nil) 
                  (reset! similarity @edit-dist)))))))
      @similarity)))
      
;; Locality-sensitive Hashing

(defn k-shingles 
  [n s]
  (let [im (into {} 
             (map-indexed (fn [idx itm] {idx itm}) (seq s)))
        shingles (map 
                   (fn [[k v]] 
                     (if (<= k (- (dec (count im)) (dec n))) 
                       (reduce str (map #(im % "") (range k (+ k n))))
                       nil))
                   im)
        shingles (filter #(not (nil? %)) shingles)]
    shingles))

(defn gen-hash-with-seeds
  [seeds]
  (map #(hasher/murmur3-32 %) seeds))

(defn hash-n-times
  [n shingles-list]
  (let [hash-fns (gen-hash-with-seeds (range n))]
    (map (fn [x] (map (fn [y] (hasher/hash->int (hasher/hash-string y x))) hash-fns)) shingles-list)))
    
(defn min-hash
  [l]
  (reduce (fn [x y] (map (fn [a b] (min a b)) x y)) l))
  
(defn partition-into-bands
  [band-size min-hashed-list]
  (partition-all band-size min-hashed-list))
  
(defn band-hash-generator
  [banded-list]
  (let [r (range -1 (unchecked-negate-int (inc (count banded-list))) -1)
        hash-fns (gen-hash-with-seeds r)
        hashed-banded-list (map 
                             (fn [x y] (hasher/hash->int (hasher/hash-string x (clojure.string/join "-" y))))
                             hash-fns banded-list)]
    hashed-banded-list))

(defn band-hash
  [s sh n b]
  (if (= s "")
    nil
    (let [shingles (k-shingles sh s)
          shingles (if (= sh 1) (map str shingles) shingles)
          hashed-n-times (hash-n-times n shingles)
          min-hashed (min-hash hashed-n-times)
          banded-list (partition-into-bands b min-hashed)
          banded-hashed (band-hash-generator banded-list)]
      banded-hashed)))

