# Canopy_microclimate_project

This project aims at answering to several objectives. 

1) Firstly, the multi-layer energy budget implemented in ORCHIDEE is tightly tied to the rest of the code, making it complicated to study the impacts of a new development on the behavior of the microclimate model. The first objective of this project is, then, to enable proper studies on the multi-layer energy budget without having to test them directly in ORCHIDEE;

2) Secondly, because of its tight ties with the rest of the ORCHIDEE code, the microclimate model resolution implemented in ORCHIDEE is complicated to follow and to change. Several improvements can be applied, especially on the way implicit coupling is currently done. This simpler version of the code permits to test those new resolution schemes more easily;

3) Thridly, the current ORCHIDEE outputs have to be treated separately from the model, making the development step, and the analysis one, two separate steps whereas they are tightly coupled. The objective of this model is, also, to use the features of python language along with the streamlit module (which facilitate the creation of online applications) to facilitate the analysis of the microclimate model;

4) Finally, this model is an extension of the current 1D model implemented in ORCHIDEE. It aims at studying the microclimate on a 2D forest with inhomogeneous properties.

To launch the application:
- Download the project;
- Open a terminal and go in the project repertory;
- Launch the following line : "streamlit run .\homepage.py";
- A local application should be opened by your navigator;
- Choose your tree shape and properties;
- Choose your forest shape and properties;
- Select your macroclimatic properties;
- Look at the other pages to see the microclimate inside the simulated forest for the macroclimatic conditions you applied. 
