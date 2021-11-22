import matlab.engine

eng = matlab.engine.start_matlab()
eng.modified_Run_Springer_Script(nargout=0)