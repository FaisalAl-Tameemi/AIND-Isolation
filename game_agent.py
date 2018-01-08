"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math
import numpy as np


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def get_unique_moves(my_moves, opp_moves):
    """
    Helper method which accepts 2 lists of moves.
    Returns moves unique to first list.
    """
    moves = []

    for move in my_moves:
        if move not in opp_moves:
            moves.append(move)

    return moves


def get_shared_moves(my_moves, opp_moves):
    shared_moves = my_moves and opp_moves
    return shared_moves


def custom_score(game, player):
    """
    An evaluation method which calculates the difference between
    the number of moves available to player and the number of moves
    available to the opponent.
    """
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    return float(len(my_moves) - len(opponent_moves))


def custom_score_2(game, player):
    """
    An evaluation method which returns a value corresponding
    to the number of unique moves available to a player.

    It can be thought of as a way to value future states where
    player is running away from the opponent.
    """
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    my_unique_moves = get_unique_moves(my_moves, opponent_moves)

    return float(len(my_unique_moves))


def custom_score_3(game, player):
    """
    An evaluation method which similar to `custom_score_1` calculates
    the difference between the moves available to each player.

    However, this function also adds a weight to the opponent's moves
    making the agent more aggressive.
    """
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    return float(len(my_moves) - (len(opponent_moves) * 1.5))


def moves_delta_exp(game, player):
    """
    A heuristic evaluation method which calculates the difference
    between available moves count between the current player and the opponent.

    This method also applies an exponential growth weight towards the opponent's
    moves count as the game progress, making the agent more aggressive as the
    number of blank spaces goes down.
    """
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")

    blank_spaces_count = len(game.get_blank_spaces())
    """
    Exponential growth: y = a(1 + r)^x
    a = 1.5
    r = (1 / blank_spaces_count)
    x = (1 / blank_spaces) + 2
    """
    weight = 1.5 * np.power((1 + (1 / blank_spaces_count)), ((1 / blank_spaces_count) + 2))
    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    return float(len(my_moves) - (len(opponent_moves) * weight))


def moves_delta_unique(game, player):
    """
    This evaluation function is a combination of custom_score_1 and custom_score_2
    where the difference between the unique moves' counts is calculated as the game state value.
    """
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    my_unique_moves = get_unique_moves(my_moves, opponent_moves)
    opponent_unique_moves = get_unique_moves(opponent_moves, my_moves)

    # return float(len(my_unique_moves))
    return float(len(my_unique_moves) - len(opponent_unique_moves))


def moves_delta_walls(game, player):
    """
    An evaluation function which, similar to custom_score_1, calculates the difference
    in the counts of moves available to each player, however, it also penalizes
    moves which are against the game board wall towards the end of the game.

    Inspired by:
    """
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")

    weight = 1
    # Penalize having corner moves late in the game
    if len(game.get_blank_spaces()) < game.width * game.height / 4.:
        weight = 4

    # game board corners
    h, w = (game.height, game.width)
    game_corners = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1)
    ]

    my_moves = game.get_legal_moves(player)
    my_corner_moves = [move for move in my_moves if move in game_corners]

    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent_corner_moves = [move for move in opponent_moves if move in game_corners]

    # calculate values based on moves while penalizing corner moves
    my_ratio = len(my_moves) - (weight * len(my_corner_moves))
    opponent_ratio = len(opponent_moves) + (weight * len(opponent_corner_moves))
    return float(my_ratio - opponent_ratio)


def moves_delta_unique_exp(game, player):
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")

    blank_spaces_count = len(game.get_blank_spaces())
    """
    Exponential growth: y = a(1 + r)^x
    a = 1.5
    r = 0.1 # decay rate
    x = (1 / blank_spaces) + 1.5
    """
    weight = 1.5 * np.power((1 + 0.1), ((1 / blank_spaces_count) + 1.5))

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    my_unique_moves = get_unique_moves(my_moves, opponent_moves)
    opponent_unique_moves = get_unique_moves(opponent_moves, my_moves)

    # return float(len(my_unique_moves))
    return float(len(my_unique_moves) - (len(opponent_unique_moves) * weight))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        top_move = {
            "move": (-1, -1),
            "score": float('-inf')
        }

        if len(legal_moves) > 0:
            for move in legal_moves:
                game_copy = game.forecast_move(move)
                game_score = self.min_value(game_copy, depth - 1, game.active_player)

                if game_score >= top_move["score"]:
                    top_move["score"] = game_score
                    top_move["move"] = move

            return top_move["move"]
        else:
            return top_move["move"]

    def min_value(self, game, depth, player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if len(game.get_legal_moves()) == 0:
            return game.utility(player)

        if depth <= 0:
            return self.score(game, player)

        score = float("inf")

        for action in game.get_legal_moves():
            game_copy = game.forecast_move(action)
            score = min(score, self.max_value(game_copy, depth - 1, player))

        return score

    def max_value(self, game, depth, player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if len(game.get_legal_moves()) == 0:
            return game.utility(player)

        if depth <= 0:
            return self.score(game, player)

        score = float("-inf")

        for action in game.get_legal_moves():
            game_copy = game.forecast_move(action)
            score = max(score, self.min_value(game_copy, depth - 1, player))

        return score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        legal_moves = game.get_legal_moves()
        top_move = (-1, -1)

        if len(legal_moves) > 0:
            top_move = legal_moves[0] # default move if any are available
        else:
            return top_move

        depth = 1

        while True:
            try:
                top_move = self.alphabeta(game, depth)
                depth += 1
            except SearchTimeout:
                return top_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        alpha_score = alpha
        beta_score = beta

        legal_moves = game.get_legal_moves()
        top_move = {"move": (-1, -1), "score": alpha_score}

        if len(legal_moves) > 0:
            top_move["move"] = legal_moves[0]

        for action in game.get_legal_moves():

            new_board = game.forecast_move(action)
            min_val = self.min_value(new_board, depth, alpha_score, beta_score, game.active_player)
            top_move["score"] = max(top_move["score"], min_val)

            if top_move["score"] > alpha_score:
                alpha_score = top_move["score"]
                top_move["move"] = action

        return top_move["move"]

    def max_value(self, game, depth, alpha, beta, player):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        new_alpha = alpha
        new_beta = beta

        if len(game.get_legal_moves(player)) == 0:
            return game.utility(player)

        score = float("-inf")

        if (depth - 1) <= 0:
            new_score = self.score(game, player)
            return new_score
        else:
            for action in game.get_legal_moves(player):
                new_game = game.forecast_move(action)

                new_score = self.min_value(
                    new_game, depth - 1, new_alpha, new_beta, player)

                score = max(score, new_score)

                if score >= new_beta:
                    return score

                new_alpha = max(new_alpha, score)

            return score

    def min_value(self, game, depth, alpha, beta, player):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        new_alpha = alpha
        new_beta = beta

        if len(game.get_legal_moves(game.get_opponent(player))) == 0:
            return game.utility(player)

        score = float("inf")

        if (depth - 1) <= 0:
            new_score = self.score(game, player)
            return new_score
        else:
            for action in game.get_legal_moves(game.get_opponent(player)):
                new_game = game.forecast_move(action)

                new_score = self.max_value(
                    new_game, depth - 1, new_alpha, new_beta, player)

                score = min(score, new_score)

                if score <= new_alpha:
                    return score

                new_beta = min(new_beta, score)

            return score
