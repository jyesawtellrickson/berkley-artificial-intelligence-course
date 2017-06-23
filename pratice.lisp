(defun sqrt-iter (a g)
  (if (goodenough? a g)
      g
      (sqrt-iter a (imgu a g))))

(defun goodenough? (a g)
  (< (abs (- a (square g))) 0.001))

(defun imgu (a g)
  (avg g (/ a g)))

(defun square (x)
  (* x x))

(defun avg (a b)
  (/ (+ a b) 2))

(defun jsqrt (a g)
  (format t "~D" (sqrt-iter a g) ))


;; --------------------------------- ;;

(defun ex13 (a)
  (pop (sortdes a)))

(defun sortdes (a)
  (sort a #'>))


