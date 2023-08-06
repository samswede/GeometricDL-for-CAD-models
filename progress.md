
# Reflection Day 1
How far did I get in 4hrs?

I only got to the point of loading files and understanding what they are composed of.
I realised that I wont be able to implement GeoCNN within the timeframe, so I went with a GNN instead.

I decided to use the features provided from the trimesh mesh, including vertex coordinates, face coordinates, and vertex normals.
Later use could include colors from an SPY file.

I will load this into a pytorch data loader.
Then I will train the GNN for classification.

I did not get as far as I expected.
I thought I could at the very least have this run overnight to load all the data into the pytorch data loader.

I also expected I would have been able to go straight into GeoCNN, but it became clear that I had to get comfortable with the object file formats first.
A GNN is a great place to start.

I think that nested networks could work for embedding composite objects but I need to find out how common CAD software stores files like that. Grouped objects is what I am interested in. I want to be able to automatically convert a grouped object into those nested graphs.
I believe that would allow for embedding, and ultimately generative design.



# IDEAS:

- Agent learns to build objects using RL. GDDPG? Should be good for this... propose coordinate changes and addition of nodes.
    - Use HRL to propose an object fingerprint subgoal state (could include acceptable bounds, too? Like a std equivalent of a VAE output)
    - Much easier primitive agent design for legal and illegal actions.

- Oh shit! I know how to synthetically make data using a diffusion style approach.
    - Start with your object, lets imagine a chair OFF / OBJ / PLS file represented as a network of vertices and faces.
    - Add gaussian noise to each node feature (coordinates, vertex normal, color properties)
    - Remove nodes in a clever way
    - It might not work that well but interesting to work backwards and have labels for each bit, to progressively get better until you reach the goal state. I mean you could chunk it like every 5 steps the subgoal is the object fingerprint of the 5th step. Could be interesting for initial pre-training?

- Classify components in an unsupervised way. Use the fingerprint of these components as subgoals.


# Reflection Day 2.

I think I should modify the project to write out exactly what I achieved within the 8hrs, and then continue until that objective is actually done and log how long it actually took me in the end.

