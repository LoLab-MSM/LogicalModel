@ECHO OFF
setlocal
set PYTHONPATH=C:\Users\James Pino\PycharmProjects\Magine;%PYTHONPATH%
kernprof -l -v magine/tests/test_network_generation.py
endlocal

