package com.rswebsite.online.util;

import com.rswebsite.online.model.Embedding;
import com.rswebsite.online.model.State;


public class Utility {
    public static Embedding parseEmbStr(String embStr){
        embStr = embStr.substring(1,embStr.length()-1);
        String[] embStrings = embStr.split(",\\s");
        Embedding emb = new Embedding();
        for (String element : embStrings) {
            emb.addDim(Float.parseFloat(element));
        }
        return emb;
    }

    public static State parseStateStr(String stateStr){
        stateStr = stateStr.substring(1,stateStr.length()-1);
        String[] stateStrings = stateStr.split(",\\s");
        State state = new State();
        for (String element : stateStrings) {
            state.addDim(Float.parseFloat(element));
        }
        return state;
    }
}