package com.rswebsite.online.model;

import java.util.ArrayList;

/**
 * State Class, contains state vector
 */
public class State {
    ArrayList<Float> stateVector;

    public State(){
        this.stateVector = new ArrayList<>();
    }

    public State(ArrayList<Float> stateVector){
        this.stateVector = stateVector;
    }

    public void addDim(Float element){
        this.stateVector.add(element);
    }

    public ArrayList<Float> getstateVector() {
        return stateVector;
    }

    public void setstateVector(ArrayList<Float> stateVector) {
        this.stateVector = stateVector;
    }
}