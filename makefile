

.PHONY: prknn


target: prknn
	python setup.py --setup build_ext --inplace

prknn:
	source set_env.sh
	rsync -rupE --progress --delete src/prknn/. prknn

clean:
	rm -rf prknn
