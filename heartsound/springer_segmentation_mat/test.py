import matlab.engine

eng = matlab.engine.start_matlab()
eng.run_Example_Springer_Script(nargout=0)