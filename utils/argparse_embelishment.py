from gettext import gettext as _
import argparse


class ArgumentParser(argparse.ArgumentParser):

    def _parse_optional(self, arg_string):
        # if it's an empty string, it was meant to be a positional
        if not arg_string:
            return None

        # if it doesn't start with a prefix, it was meant to be positional
        if not arg_string[0] in self.prefix_chars:
            return None

        # if it's just a single character, it was meant to be positional
        if len(arg_string) == 1:
            return None

        option_tuples = self._get_option_tuples(arg_string)

        # if multiple actions match, the option string was ambiguous
        if len(option_tuples) > 1:
            options = ', '.join([option_string
                for action, option_string, explicit_arg in option_tuples])
            args = {'option': arg_string, 'matches': options}
            msg = _('ambiguous option: %(option)s could match %(matches)s')
            self.error(msg % args)

        # if exactly one action matched, this segmentation is good,
        # so return the parsed action
        elif len(option_tuples) == 1:
            option_tuple, = option_tuples
            return option_tuple

        # if it was not found as an option, but it looks like a negative
        # number, it was meant to be positional
        # unless there are negative-number-like options
        if self._negative_number_matcher.match(arg_string):
            if not self._has_negative_number_optionals:
                return None

        # if it contains a space, it was meant to be a positional
        if ' ' in arg_string:
            return None

        # it was meant to be an optional but there is no such option
        # in this parser (though it might be a valid option in a subparser)
        return None, arg_string, None

    def _get_option_tuples(self, option_string):
        result = []

        if '=' in option_string:
            option_prefix, explicit_arg = option_string.split('=', 1)
        else:
            option_prefix = option_string
            explicit_arg = None
        if option_prefix in self._option_string_actions:
            action = self._option_string_actions[option_prefix]
            tup = action, option_prefix, explicit_arg
            result.append(tup)
        else:  # imperfect match
            chars = self.prefix_chars
            if option_string[0] in chars and option_string[1] not in chars:
                # short option: if single character, can be concatenated with arguments
                short_option_prefix = option_string[:2]
                short_explicit_arg = option_string[2:]
                if short_option_prefix in self._option_string_actions:
                    action = self._option_string_actions[short_option_prefix]
                    tup = action, short_option_prefix, short_explicit_arg
                    result.append(tup)

            underscored = {k.replace('-', '_'): k for k in self._option_string_actions}
            option_prefix = option_prefix.replace('-', '_')
            if option_prefix in underscored:
                action = self._option_string_actions[underscored[option_prefix]]
                tup = action, underscored[option_prefix], explicit_arg
                result.append(tup)
            elif self.allow_abbrev:
                    for option_string in underscored:
                        if option_string.startswith(option_prefix):
                            action = self._option_string_actions[underscored[option_string]]
                            tup = action, underscored[option_string], explicit_arg
                            result.append(tup)

        # return the collected option tuples
        return result