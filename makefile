

.PHONY: prknn


target: prknn
	python setup.py --setup build_ext --inplace

prknn:
	rsync -rupE --delete src/prknn/. prknn

clean:
	rm -rf prknn
