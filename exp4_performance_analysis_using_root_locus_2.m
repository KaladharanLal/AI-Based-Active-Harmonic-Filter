clear all;
clc;
s = tf('s');
sys = (1/((s*(s+1)*(s+2))));
controlSystemDesigner(sys);