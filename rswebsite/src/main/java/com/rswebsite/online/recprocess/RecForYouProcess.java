package com.rswebsite.online.recprocess;

import com.rswebsite.online.datamanager.DataManager;
import com.rswebsite.online.datamanager.User;
import com.rswebsite.online.datamanager.Movie;
import com.rswebsite.online.datamanager.RedisClient;
import com.rswebsite.online.model.Embedding;
import com.rswebsite.online.util.Config;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;

import static com.rswebsite.online.util.HttpClient.asyncSinglePostRequest;

/**
 * Recommendation process
 */

public class RecForYouProcess {

    /**
     * get recommendation movie list
     * @param userId input user id
     * @param size  size of similar items
     * @param model model used for calculating similarity
     * @return  list of similar movies
     */
    public static List<Movie> getRecList(int userId, int size, String model){
        User user = DataManager.getInstance().getUserById(userId);
        if (null == user){
            return new ArrayList<>();
        }
        final int CANDIDATE_SIZE = 1600;
        List<Movie> candidates = DataManager.getInstance().getMovies(CANDIDATE_SIZE, "rating");

        if (Config.IS_LOAD_USER_FEATURE_FROM_REDIS){
            String userFeaturesKey = "uf:" + userId;
            Map<String, String> userFeatures = RedisClient.getInstance().hgetAll(userFeaturesKey);
            if (null != userFeatures){
                user.setUserFeatures(userFeatures);
            }
        }

        List<Movie> rankedList = ranker(user, candidates, model);

        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    /**
     * rank candidates
     * @param user    input user
     * @param candidates    movie candidates
     * @param model     model name used for ranking
     * @return  ranked movie list
     */
    public static List<Movie> ranker(User user, List<Movie> candidates, String model){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();

        switch (model){
            case "rl":
                callRLTFServing(user, candidates, candidateScoreMap);
                break;
            default:
                //default ranking in candidate set
                for (int i = 0 ; i < candidates.size(); i++){
                    candidateScoreMap.put(candidates.get(i), (double)(candidates.size() - i));
                }
        }

        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    private static void callRLTFServing(User user, List<Movie> candidates, HashMap<Movie, Double> candidateScoreMap) {
        if (null == user || null == candidates || candidates.size() == 0){
            return;
        }

        JSONObject instance = new JSONObject();
        instance.put("input_1",user.getState().getstateVector());
        JSONArray instances = new JSONArray();
        instances.put(instance);
        JSONObject instancesRoot = new JSONObject();
        instancesRoot.put("instances",instances);
        String prediction = asyncSinglePostRequest("http://localhost:8501/v1/models/recmodel:predict", instancesRoot.toString());
        JSONObject predictionsObject = new JSONObject(prediction);
        JSONArray scores = predictionsObject.getJSONArray("predictions");
        System.out.println("send user" + user.getUserId() + " request to tf serving.");
        ArrayList<Float> embVector = new ArrayList<>();

        for(int i =0;i <100;++i){
            embVector.add((float)scores.getJSONArray(0).getDouble(i));
        }
        Embedding e = new Embedding(embVector);
        for (Movie candidate : candidates){
            double similarity = candidate.getEmb().calculateSimilarity(e);
            candidateScoreMap.put(candidate, similarity);
        }
    }

    /**
     * call TenserFlow serving to get the NeuralCF model inference result
     * @param user              input user
     * @param candidates        candidate movies
     * @param candidateScoreMap save prediction score into the score map
     */
    public static void callNeuralCFTFServing(User user, List<Movie> candidates, HashMap<Movie, Double> candidateScoreMap){
        if (null == user || null == candidates || candidates.size() == 0){
            return;
        }

        JSONArray instances = new JSONArray();
        for (Movie m : candidates){
            JSONObject instance = new JSONObject();
            instance.put("userId", user.getUserId());
            instance.put("movieId", m.getMovieId());
            instances.put(instance);
        }

        JSONObject instancesRoot = new JSONObject();
        instancesRoot.put("instances", instances);
        //need to confirm the tf serving end point
        String predictionScores = asyncSinglePostRequest("http://localhost:8501/v1/models/recmodel:predict", instancesRoot.toString());
        System.out.println("send user" + user.getUserId() + " request to tf serving.");
        JSONObject predictionsObject = new JSONObject(predictionScores);
        JSONArray scores = predictionsObject.getJSONArray("predictions");
        for (int i = 0 ; i < candidates.size(); i++){
            candidateScoreMap.put(candidates.get(i), scores.getJSONArray(i).getDouble(0));
        }
    }
}