using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System;
using System.Collections.Generic;

namespace NeuralNet
{
    /// <summary>
    /// This is the main type for your game.
    /// </summary>
    public class NeuralNet : Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;
        Network.networkData nd = new Network.networkData();
        IrisData id = new IrisData();
        Graph graph;
        List<Tuple<float, Color>> graphPoints = new List<Tuple<float, Color>>();

        bool keyUp = false;
        bool keyDown = false;
        
        uint samples = 500;
        double learnRate = 0.3;
        double momentum = 0.8;
        uint graphWidth = 250;

        public void initNet()
        {
            //Make a new Layers
            nd.layers = new Network.Layers(nd.training.inputs.cols, nd.training.outputs.cols);

            //Create some new layers for the for the network 
            nd.layers.create(10, 4, 0.5, ActivateMode.Sigmoid);
            nd.layers.create(10, 4, 0.5, ActivateMode.Sigmoid);
            nd.layers.create(3, 1, 0.5, ActivateMode.Sigmoid);
        }


        public NeuralNet()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
        }

        /// <summary>
        /// Allows the game to perform any initialization it needs to before starting to run.
        /// This is where it can query for any required services and load any non-graphic
        /// related content.  Calling base.Initialize will enumerate through any components
        /// and initialize them as well.
        /// </summary>
        protected override void Initialize()
        {
            // TODO: Add your initialization logic here
            
            //NeuralNetwork nn = new NeuralNetwork();
            nd.inputCount = id.irisTest.inputs.cols; //4;
            nd.outputCount = id.irisTest.outputs.cols; //3;
            nd.training = id.irisTest;
            nd.validation = id.irisTest;
            nd.test = id.irisTest;
            
            for (int i = 0; i < graphWidth; i++)
            {
                Tuple<float, Color> graphPoint = new Tuple<float, Color>(0.0f, new Color(255, 255, 255));
                graphPoints.Add(graphPoint);
            }

            initNet();

            graph = new Graph(graphics.GraphicsDevice, new Point(700, 500));
            graph.Position = new Vector2(0, 500);
            graph.MaxValue = 1;
            graph.Type = Graph.GraphType.Fill;

            base.Initialize();
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
            // Create a new SpriteBatch, which can be used to draw textures.
            spriteBatch = new SpriteBatch(GraphicsDevice);

            // TODO: use this.Content to load your game content here
        }

        /// <summary>
        /// UnloadContent will be called once per game and is the place to unload
        /// game-specific content.
        /// </summary>
        protected override void UnloadContent()
        {
            // TODO: Unload any non ContentManager content here
        }

        /// <summary>
        /// Allows the game to run logic such as updating the world,
        /// checking for collisions, gathering input, and playing audio.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();
            if (Keyboard.GetState().IsKeyDown(Keys.R))
                initNet();
            if (Keyboard.GetState().IsKeyDown(Keys.Up))
                keyUp = true;
            if (Keyboard.GetState().IsKeyUp(Keys.Up) && keyUp)
            {
                samples += 1;
                keyUp = false;
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Down))
                keyDown = true;
            if (Keyboard.GetState().IsKeyUp(Keys.Down) && keyDown)
            {
                if (samples > 1) samples -= 1;
                keyDown = false;
            }

            //Train the network
            Network.train(ref nd, samples, learnRate, momentum, ref graphPoints, graphWidth);

            // TODO: Add your update logic here

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.CornflowerBlue);

            // TODO: Add your drawing code here

            graph.Draw(graphPoints);

            base.Draw(gameTime);
        }
    }
}
