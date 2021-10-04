from __future__ import annotations
from typing import List, Optional, Dict, Text, Any

from rasa.core.turns.turn import Actor, DefinedTurn, Turn
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.events import ActionExecuted, Event, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    PREVIOUS_ACTION,
)


class StatefulTurn(DefinedTurn):
    """A wrapper for the classic 'State'.

    This kind of state is no 'dialogue turn'. It is a representation of the user and
    bot turns as "perceived" by the bot at the point where it has to decide the next
    action.

    What do "stateful turns" capture?

        At the point where the bot has to decide the next action, we are given a
        sequence of events that either ends with a user utterance or a bot action,
        which is not an `ActionListen` (because in this case we wait for the user
        input). Moreover, we are given the sequence of events that describes the
        inner workings of the bot up to the point where it has to decide on the
        next action.

        Hence, given a sequence of events, we create a "stateful turn" for
        every sequence of events that (starts with the first event and) is followed
        by an `ActionExecuted` event (but does not include that last event).

    Bot or User Turn?

        In case the `ActionExecuted` is an `ActionListen`, we know that our
        "stateful turn" captures a user utterance
        In the special case where an ActionListen is the last state and the only
        event captured in the turn, we still attribute this turn to the user.

    Why is it "Stateful"?

        Assume we want to generate a state from a sequence of events. This sequence
        includes the very first recorded event up to an event that is followed by
        an `ActionExecuted`.
        We only attribute all events that where not included in the previous turn to
        the new turn that we want to create.
        However, we also store the last bot action and last user utterance plus
        the context, which consists of the current "active loop" and current status of
        slots (which are pre-featurized) in the "stateful turn".
        Observe that the latter includes information about all events so far, not just
        the ones attributed to the current turn.

    Some events are Missing

        This kind of turn is a representation that helps the bot to decide the next
        action. Hence, the sequence of events from which we derive "stateful turns"
        are the result of applying all tracker events once, i.e. the bot will not
        see events like `UserUtteranceReverted` and `SessionRestarted`.

    Ignoring Rule Turns

        If we ignore rule turns, then we do not generate a turn if the respective
        `ActionExecuted` event is the result of the application of a rule.

        Moreover, if we ignore rule turns, we ignore all "rule data", i.e.
        loops and slots that only appear in rules but not in stories.
    """

    def __init__(
        self, credentials: object, actor: Actor, events: List[Event], state: State
    ) -> None:
        """Private constructor for stateful turns. Use `parse` to generate these turns.

        Args:
            credentials: special object that allows you to use this method
            actor: describes whose turn it is (user or bot)
            events: a sequence of events belonging to the `actor`
            state: a description of the last user and bot utterance, the current slot
               values and information about any active loop
        """
        super().__init__(credentials=credentials, actor=actor, events=events)
        self.state = state

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StatefulTurn):
            return NotImplemented
        return self.state == other.state

    def __hash__(self) -> int:
        frozen_states = DialogueStateTracker.freeze_current_state(self.state)
        return hash(frozen_states)

    @classmethod
    def parse(
        cls,
        tracker: DialogueStateTracker,
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[StatefulTurn]:
        """Returns all `RuleStates` from the trackers event history.

        Args:
            tracker: Dialogue state tracker containing the dialogue so far.
            domain: the common domain
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_rule_only_turns: If set to `True`, we ignore `ActionExecuted` events
               that are the result of the application of a rule and hence we also do
               not create the corresponding turn.
            rule_only_data: Slots and loops, which only occur in rules but not in
              stories.

        Return:
            ...
        """
        turns = []
        turns.append(
            StatefulTurn(
                credentials=DefinedTurn._credentials,
                actor=Actor.BOT,
                state={},
                events=[],
            )
        )

        applied_events = tracker.applied_events()

        replay_tracker = tracker.init_copy()
        turn_actor = Actor.BOT
        turn_events = []

        if ignore_rule_only_turns:
            last_turn_was_hidden = False
            last_non_hidden_action_executed = None

        for idx, current_event in enumerate(applied_events):

            # Apply the event
            replay_tracker.update(current_event)
            turn_events.append(current_event)

            # Update actor information
            if isinstance(current_event, UserUttered):
                if not ignore_rule_only_turns and turn_actor == Actor.USER:
                    raise RuntimeError(
                        "Found a `UserUttered` event that was possibly followed "
                        "by some events that were *not* `ActionExecutedEvents` "
                        "before encountering the next `UserUttered` event, "
                        "event though we are not ignoring rule turns."
                    )
                turn_actor = Actor.USER

            # Generate a turn ...
            if idx == len(applied_events) - 1 or isinstance(
                applied_events[idx + 1], ActionExecuted
            ):

                #  .... unless the next (action executed) event is hidden
                if ignore_rule_only_turns:

                    # Observe that the last turn was hidden if and only if the
                    # *current* event is hidden. However, the tracker is unaware of
                    # this and will return this current hidden event after we've
                    # updated it with the current event. Hence, we have to remember:
                    if not last_turn_was_hidden:
                        last_non_hidden_action_executed = replay_tracker.latest_action

                    # The next turn is hidden if the next event (`ActionExecuted`) is
                    # hidden, which happens if it is an has been generated by a rule
                    # - unless it is a follow up action or a happy loop prediction.
                    if idx < len(applied_events) - 1:
                        if not replay_tracker.followup_action and (
                            not replay_tracker.latest_action_name
                            == replay_tracker.active_loop_name
                        ):
                            next_event = applied_events[idx + 1]
                            last_turn_was_hidden = next_event.hide_rule_turn
                            if last_turn_was_hidden:
                                continue

                # Create a state first
                turn_state = domain.get_active_state(
                    replay_tracker, omit_unset_slots=omit_unset_slots
                )
                if ignore_rule_only_turns:
                    # Clean state from only rule features
                    domain._remove_rule_only_features(turn_state, rule_only_data)
                    # Make sure user input is the same as for previous state
                    # for non action_listen turns # FIXME: is this needed?
                    domain._substitute_rule_only_user_input(turn_state, turns[-1].state)
                    # If we skipped the creation of a turn because the next
                    # `ActionExecuted` event was hidden, then we also do not want to
                    # use this event as a 'previous action' (which would be returned
                    # by the `tracker.latest_action` at this point). Instead:
                    if last_non_hidden_action_executed:
                        turn_state[PREVIOUS_ACTION] = last_non_hidden_action_executed
                    # Cleanup again
                    cls.remove_empty_sub_states(turn_state)

                # Special Case: The turn just captures an `ActionListen`. In this
                # case, we attribute the turn to the user, even though the turn does
                # not capture a new utterance to handle `ActionListen` consistently.
                if (
                    turn_actor == Actor.BOT
                    and turn_events[0].action_name == ACTION_LISTEN_NAME
                ):
                    turn_actor = Actor.USER

                # Create the turn
                turn = StatefulTurn(
                    credentials=DefinedTurn._credentials,
                    actor=turn_actor,
                    state=turn_state,
                    events=turn_events,
                )
                turns.append(turn)

                # Reset information about current turn.
                turn_actor = Actor.BOT
                turn_events = []

        return turns

    @classmethod
    def remove_empty_sub_states(cls, state: State) -> None:
        empty = [
            sub_state_name
            for sub_state_name, sub_state in state.items()
            if not sub_state
        ]
        for sub_state_name in empty:
            del state[sub_state_name]
