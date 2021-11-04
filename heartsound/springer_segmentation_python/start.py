import matlab.engine

eng = matlab.engine.start_matlab()
for i in range(0, 5):
    print(eng.test(10))