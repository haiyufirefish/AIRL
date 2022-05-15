package com.rswebsite.online;

import com.rswebsite.online.datamanager.DataManager;
import com.rswebsite.online.service.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;

import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;

public class RecServer {
    public static void main(String[] args) throws Exception {
        new RecServer().run();
    }

    //recsys server port number
    private static final int DEFAULT_PORT = 6016;

    public void run() throws Exception {

        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {
        }

        //set ip and port number
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        //get index.html path
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null) {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        //set index.html as the root page
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$", "/"));
        System.out.printf("Web Root URI: %s%n", webRootUri.getPath());

        //load all the data to DataManager
        DataManager.getInstance().loadData(webRootUri.getPath() + "sampledata2/movies_100k.csv",
                webRootUri.getPath() + "sampledata2/links.csv", webRootUri.getPath() + "sampledata2/ratings_100k.csv",
                webRootUri.getPath() + "modeldata/item_embeddings2.csv",
                webRootUri.getPath()+"modeldata/states2.csv",
                "i2vEmb", "uEmb");

        //create server context
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        context.setBaseResource(Resource.newResource(webRootUri));
        context.setWelcomeFiles(new String[]{"index.html"});
        context.getMimeTypes().addMimeMapping("txt", "text/plain;charset=utf-8");

        //bind services with different servlets
        context.addServlet(DefaultServlet.class, "/");
        context.addServlet(new ServletHolder(new MovieService()), "/getmovie");
        context.addServlet(new ServletHolder(new UserService()), "/getuser");
        context.addServlet(new ServletHolder(new RecommendationService()), "/getrecommendation");
        context.addServlet(new ServletHolder(new RecForYouService()), "/getrecforyou");

        //set url handler
        server.setHandler(context);
        System.out.println("RecSys Server has started.");

        //start Server
        server.start();
        server.join();
    }
}