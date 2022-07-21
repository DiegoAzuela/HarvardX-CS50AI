"""
SOURCE
    https://cs50.harvard.edu/ai/2020/projects/0/degrees/
    
SOLVED BY
    Diego Arnoldo Azuela Rosas

BACKGROUND
    According to the Six Degrees of Kevin Bacon game, anyone in the Hollywood film industry can be connected to Kevin Bacon within six steps, where each step consists of finding a film that two actors both starred in.
    In this problem, we’re interested in finding the shortest path between any two actors by choosing a sequence of movies that connects them. For example, the shortest path between Jennifer Lawrence and Tom Hanks is 2: Jennifer Lawrence is connected to Kevin Bacon by both starring in “X-Men: First Class,” and Kevin Bacon is connected to Tom Hanks by both starring in “Apollo 13.”
    We can frame this as a search problem: our states are people. Our actions are movies, which take us from one actor to another (it’s true that a movie could take us to multiple different actors, but that’s okay for this problem). Our initial state and goal state are defined by the two people we’re trying to connect. By using breadth-first search, we can find the shortest path from one actor to another.

UNDERSTANDING
    The distribution code contains two sets of CSV data files: one set in the large directory and one set in the small directory. Each contains files with the same names, and the same structure, but small is a much smaller dataset for ease of testing and experimentation.
    Each dataset consists of three CSV files. A CSV file, if unfamiliar, is just a way of organizing data in a text-based format: each row corresponds to one data entry, with commas in the row separating the values for that entry.
    Open up small/people.csv. You’ll see that each person has a unique id, corresponding with their id in IMDb’s database. They also have a name, and a birth year.
    Next, open up small/movies.csv. You’ll see here that each movie also has a unique id, in addition to a title and the year in which the movie was released.
    Now, open up small/stars.csv. This file establishes a relationship between the people in people.csv and the movies in movies.csv. Each row is a pair of a person_id value and movie_id value. The first row (ignoring the header), for example, states that the person with id 102 starred in the movie with id 104257. Checking that against people.csv and movies.csv, you’ll find that this line is saying that Kevin Bacon starred in the movie “A Few Good Men.”
    Next, take a look at degrees.py. At the top, several data structures are defined to store information from the CSV files. The names dictionary is a way to look up a person by their name: it maps names to a set of corresponding ids (because it’s possible that multiple actors have the same name). The people dictionary maps each person’s id to another dictionary with values for the person’s name, birth year, and the set of all the movies they have starred in. And the movies dictionary maps each movie’s id to another dictionary with values for that movie’s title, release year, and the set of all the movie’s stars. The load_data function loads data from the CSV files into these data structures.
    The main function in this program first loads data into memory (the directory from which the data is loaded can be specified by a command-line argument). Then, the function prompts the user to type in two names. The person_id_for_name function retrieves the id for any person (and handles prompting the user to clarify, in the event that multiple people have the same name). The function then calls the shortest_path function to compute the shortest path between the two people, and prints out the path.
    The shortest_path function, however, is left unimplemented. That’s where you come in!

SPECIFICATION
    Complete the implementation of the shortest_path function such that it returns the shortest path from the person with id source to the person with the id target.
        Assuming there is a path from the source to the target, your function should return a list, where each list item is the next (movie_id, person_id) pair in the path from the source to the target. Each pair should be a tuple of two strings.
            For example, if the return value of shortest_path were [(1, 2), (3, 4)], that would mean that the source starred in movie 1 with person 2, person 2 starred in movie 3 with person 4, and person 4 is the target.
        If there are multiple paths of minimum length from the source to the target, your function can return any of them.
        If there is no possible path between two actors, your function should return None.
        You may call the neighbors_for_person function, which accepts a person’s id as input, and returns a set of (movie_id, person_id) pairs for all people who starred in a movie with a given person.
    You should not modify anything else in the file other than the shortest_path function, though you may write additional functions and/or import other Python standard library modules.

HINTS
    While the implementation of search in lecture checks for a goal when a node is popped off the frontier, you can improve the efficiency of your search by checking for a goal as nodes are added to the frontier: if you detect a goal node, no need to add it to the frontier, you can simply return the solution immediately.
    You’re welcome to borrow and adapt any code from the lecture examples. We’ve already provided you with a file util.py that contains the lecture implementations for Node, StackFrontier, and QueueFrontier, which you’re welcome to use (and modify if you’d like).
"""


import time

import csv
import sys

from util import Node, StackFrontier, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass


def main():
    t0 = time.time()
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    load_data(directory)
    print("Data loaded.")

    source = person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    path = shortest_path(source, target)

    if path is None:
        print("Not connected.")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        path = [(None, source)] + path
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")
            t1 = time.time()
            print(f"Time lapse", t1-t0)


def shortest_path(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.

    Improvements: Current solution does not take into account path_cost, simply takes into account first viable solution
    """
    # TODO

    # NOTES: What is required? If this is a game: we need the frontier, number of nodes, number of nodes explored, current node, path cost
    
    # Keep track of the Path Cost ('internal use') and Path Cost Result ('displayable')
    path_cost, path_cost_result = [],[]
    # Create a set of reviewed actors thus far
    actors_reviewed = set()
    # Initialize Node with source as head of the tree
    tree_start = Node(state = source, parent = None, action = None)
    # StackFrontier is a parent class to QueueFrontier
    frontier = QueueFrontier()
    frontier.add(tree_start)

    while True:
        # 1. If the frontier is empty, _Stop_ there is no solution to the problem.
        if frontier.empty():
            return None

        # 2. Remove a node from the frontier. This is the node that will be considered. -This actor will be updated as 'reviewed'-
        node = frontier.remove()
        actors_reviewed.add(node.state)
        path_cost.append(node.action) 
        # print(len(path_cost))

        # -Use the function 'neighbors_for_person' to find possible neighbors-
        neighbors = neighbors_for_person(node.state)

        for movie, actor in neighbors:
            if (frontier.contains_state(actor) == False) and (actor not in actors_reviewed):
                child = Node(state = actor, parent = node, action = movie)
                # 3. If the node contains the goal state, Return the solution _Stop_. --Returns the shortest list of (movie_id, person_id) pairs--
                if child.state == target:
                    node = child
                    path_cost_result.append((node.action, node.state))  
                    # -Must correct the way the list is displayed-
                    path_cost_result.reverse() 
                    print(path_cost_result)
                    return path_cost_result
                # 4. Expand the node (find all the new nodes that could be reached from this node), and add resulting nodes to the frontier. Add the current node to the explored set
                frontier.add(child)


def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors


if __name__ == "__main__":
    main()
