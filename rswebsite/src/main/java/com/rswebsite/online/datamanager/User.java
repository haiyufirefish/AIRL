package com.rswebsite.online.datamanager;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.rswebsite.online.model.State;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * User class, contains attributes loaded from movielens ratings.csv
 */
public class User {
    int userId;
    double averageRating = 0;
    double highestRating = 0;
    double lowestRating = 5.0;
    int ratingCount = 0;

    @JsonSerialize(using = RatingListSerializer.class)
    List<Rating> ratings;

//    state of the movie
    @JsonIgnore
    State state;

    @JsonIgnore
    Map<String, String> userFeatures;

    public User(){
        this.ratings = new ArrayList<>();
        this.state = null;
        this.userFeatures = null;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public List<Rating> getRatings() {
        return ratings;
    }

    public void setRatings(List<Rating> ratings) {
        this.ratings = ratings;
    }

    public void addRating(Rating rating) {
        this.ratings.add(rating);
        this.averageRating = (this.averageRating * ratingCount + rating.getScore()) / (ratingCount + 1);
        if (rating.getScore() > highestRating){
            highestRating = rating.getScore();
        }

        if (rating.getScore() < lowestRating){
            lowestRating = rating.getScore();
        }

        ratingCount++;
    }


    public State getState() {
        return state;
    }

    public void setState(State state) {
        this.state = state;
    }

    public Map<String, String> getUserFeatures() {
        return userFeatures;
    }

    public void setUserFeatures(Map<String, String> userFeatures) {
        this.userFeatures = userFeatures;
    }
}