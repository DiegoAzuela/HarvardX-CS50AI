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
    directory = sys.argv[1] if len(sys.argv) == 2 else "small"

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
