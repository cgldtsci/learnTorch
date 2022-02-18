import yaml

class cwrap(object):

    DEFAULT_PLUGIN_CLASSES = [
        # ArgcountChecker, ConstantArguments, OptionalArguments, ArgumentReferences, BeforeCall, ReturnArguments
    ]

    def __init__(self, source, destination=None, plugins=[], default_plugins=True):
        if destination is None:
            destination = source.replace('.cwrap', '.cpp')

        self.plugins = plugins
        if default_plugins:
            defaults = [cls() for cls in self.DEFAULT_PLUGIN_CLASSES]
            self.plugins = defaults + self.plugins

        # for plugin in self.plugins:
        #     plugin.initialize(self)

        with open(source, 'r') as f:
            declarations = f.read()

        wrapper = self.wrap_declarations(declarations)

    def wrap_declarations(self, declarations):
        lines = declarations.split('\n')
        declaration_lines = []
        output = []
        in_declaration = False

        for line in lines:
            if line == '[[':
                declaration_lines = []
                in_declaration = True
            elif line == ']]':
                in_declaration = False
                declaration = yaml.safe_load('\n'.join(declaration_lines))
                self.set_declaration_defaults(declaration)
                print(declaration)

                # Pass declaration in a list - maybe some plugins want to add
                # multiple wrappers
                declarations = [declaration]
                # for plugin in self.plugins:
                #     declarations = plugin.process_declarations(declarations)
                # Generate wrappers for all declarations and append them to
                # the output
                for declaration in declarations:
                    pass
                    # wrapper = self.generate_wrapper(declaration)

            elif in_declaration:
                declaration_lines.append(line)
            else:
                output.append(line)

        return '\n'.join(output)

    def set_declaration_defaults(self, declaration):
        declaration.setdefault('arguments', [])
        declaration.setdefault('return', 'void')
        if not 'cname' in declaration:
            declaration['cname'] = declaration['name']
        # Simulate multiple dispatch, even if it's not necessary
        if not 'options' in declaration:
            declaration['options'] = [{'arguments': declaration['arguments']}]
            del declaration['arguments']
        # Parse arguments (some of them can be strings)
        for option in declaration['options']:
            option['arguments'] = self.parse_arguments(option['arguments'])
        # Propagate defaults from declaration to options
        for option in declaration['options']:
            for k, v in declaration.items():
                if k != 'name' and k != 'options':
                    option.setdefault(k, v)

    def parse_arguments(self, args):
        new_args = []
        for arg in args:
            # Simple arg declaration of form "<type> <name>"
            if isinstance(arg, str):
                t, _, name = arg.partition(' ')
                new_args.append({'type': t, 'name': name})
            elif isinstance(arg, dict):
                if 'arg' in arg:
                    arg['type'], _, arg['name'] = arg['arg'].partition(' ')
                    del arg['arg']
                new_args.append(arg)
            else:
                assert False
        return new_args