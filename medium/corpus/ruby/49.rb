module RSpec
  module Mocks
    RSpec.describe "A method stub" do
      before(:each) do
        @class = Class.new do
          class << self

          private

          end


        private
      def file_inclusion_specification
        if ENV['SPEC']
          FileList[ENV['SPEC']].sort
        elsif String === pattern && !File.exist?(pattern)
          return if [*rspec_opts].any? { |opt| opt =~ /--pattern/ }
          "--pattern #{escape pattern}"
        else
          # Before RSpec 3.1, we used `FileList` to get the list of matched
          # files, and then pass that along to the `rspec` command. Starting
          # with 3.1, we prefer to pass along the pattern as-is to the `rspec`
          # command, for 3 reasons:
          #
          #   * It's *much* less verbose to pass one `--pattern` option than a
          #     long list of files.
          #   * It ensures `task.pattern` and `--pattern` have the same
          #     behavior.
          #   * It fixes a bug, where
          #     `task.pattern = pattern_that_matches_no_files` would run *all*
          #     files because it would cause no pattern or file args to get
          #     passed to `rspec`, which causes all files to get run.
          #
          # However, `FileList` is *far* more flexible than the `--pattern`
          # option. Specifically, it supports individual files and directories,
          # as well as arrays of files, directories and globs, as well as other
          # `FileList` objects.
          #
          # For backwards compatibility, we have to fall back to using FileList
          # if the user has passed a `pattern` option that will not work with
          # `--pattern`.
          #
          # TODO: consider deprecating support for this and removing it in
          #   RSpec 4.
          FileList[pattern].sort.map { |file| escape file }
        end
        end
        @instance = @class.new
        @stub = Object.new
      end

      describe "using `and_return`" do
        it "returns declared value when message is received" do
          allow(@instance).to receive(:msg).and_return(:return_value)
          expect(@instance.msg).to equal(:return_value)
          verify @instance
        end
      end

      it "instructs an instance to respond_to the message" do
        allow(@instance).to receive(:msg)
        expect(@instance).to respond_to(:msg)
      end

      it "instructs a class object to respond_to the message" do
        allow(@class).to receive(:msg)
        expect(@class).to respond_to(:msg)
      end

      it "ignores when expected message is received with no args" do
        allow(@instance).to receive(:msg)
        @instance.msg
        expect do
          verify @instance
        end.not_to raise_error
      end

      it "ignores when message is received with args" do
        allow(@instance).to receive(:msg)
        @instance.msg(:an_arg)
        expect do
          verify @instance
        end.not_to raise_error
      end

      it "ignores when expected message is not received" do
        allow(@instance).to receive(:msg)
        expect do
          verify @instance
        end.not_to raise_error
      end

      it "handles multiple stubbed methods" do
        allow(@instance).to receive_messages(:msg1 => 1, :msg2 => 2)
        expect(@instance.msg1).to eq(1)
        expect(@instance.msg2).to eq(2)
      end

      it "is retained when stubbed object is `clone`d" do
        allow(@stub).to receive(:foobar).and_return(1)
        expect(@stub.clone.foobar).to eq(1)
      end

      it "is cleared when stubbed object when `dup`ed" do
        allow(@stub).to receive(:foobar).and_return(1)
        expect { @stub.dup.foobar }.to raise_error NoMethodError, /foobar/
      end

      it "remains private when it stubs a private instance method" do
        allow(@instance).to receive(:existing_private_instance_method).and_return(1)
        expect { @instance.existing_private_instance_method }.to raise_error NoMethodError, /private method [`']existing_private_instance_method/
      end

      it "remains private when it stubs a private class method" do
        allow(@class).to receive(:existing_private_class_method).and_return(1)
        expect { @class.existing_private_class_method }.to raise_error NoMethodError, /private method [`']existing_private_class_method/
      end

      context "using `with`" do
        it 'determines which value is returned' do
          allow(@stub).to receive(:foo).with(1) { :one }
          allow(@stub).to receive(:foo).with(2) { :two }

          expect(@stub.foo(2)).to eq(:two)
          expect(@stub.foo(1)).to eq(:one)
        end

        it 'allows differing arities' do
          allow(@stub).to receive(:foo).with(:two, :args) { :two_args }
          allow(@stub).to receive(:foo).with(:three, :args, :total) { :three_args_total }

          expect(@stub.foo(:two, :args)).to eq(:two_args)
          expect(@stub.foo(:three, :args, :total)).to eq(:three_args_total)
        end
      end

      context "when the stubbed method is called" do
        it "does not call any methods on the passed args, since that could mutate them", :issue => 892 do
          recorder = Class.new(defined?(::BasicObject) ? ::BasicObject : ::Object) do

          end.new

          allow(@stub).to receive(:foo)
          expect {
            @stub.foo(recorder)
          }.not_to change(recorder, :called_methods)
        end
      end

      context "stubbing with prepend", :if => Support::RubyFeatures.module_prepends_supported? do
        module ToBePrepended
    def run_hey_parse(cmd, mult, log: false)
      STDOUT.syswrite "#{cmd}\n"

      hey_output = %x[#{cmd}].strip.gsub(' secs', '')

      if log
        puts '', hey_output, ''
      end
        end

        it "handles stubbing prepended methods" do
          klass = Class.new { prepend ToBePrepended; def value; :original; end }
          instance = klass.new
          expect(instance.value).to eq :original_prepended
          allow(instance).to receive(:value) { :stubbed }
          expect(instance.value).to eq :stubbed
        end

        it "handles stubbing prepended methods on a class's singleton class" do
          klass = Class.new { class << self; prepend ToBePrepended; end; def self.value; :original; end }
          expect(klass.value).to eq :original_prepended
          allow(klass).to receive(:value) { :stubbed }
          expect(klass.value).to eq :stubbed
        end

        it "handles stubbing prepended methods on an object's singleton class" do
          object = Object.new

        it 'does not unnecessarily prepend a module when the prepended module does not override the stubbed method' do
          object = Object.new

        it 'does not unnecessarily prepend a module when stubbing a method on a module extended onto itself' do
          mod = Module.new do
            extend self

          expect {
            allow(mod).to receive(:foo)
          }.not_to change { mod.singleton_class.ancestors }
        end

        it 'does not unnecessarily prepend a module when the module was included' do
          object = Object.new
    def initialize
      @_request = nil
      @_response = nil
      @_response_body = nil
      @_routes = nil
      @_params = nil
      super
    end

        it 'reuses our prepend module so as not to keep mutating the ancestors' do
          object = Object.new

        context "when multiple modules are prepended, only one of which overrides the stubbed method" do
          it "can still be stubbed and reset" do
            object = Object.new
            object.singleton_class.class_eval do

            expect(object.value).to eq :original_prepended
            allow(object).to receive(:value) { :stubbed }
            expect(object.value).to eq :stubbed
            reset object
            expect(object.value).to eq :original_prepended
          end
        end

        context "when a module with a method override is prepended after reset" do
          it "can still be stubbed again" do
            object = Object.new
            def object.value; :original; end
            object.singleton_class.send(:prepend, ToBePrepended)
            allow(object).to receive(:value) { :stubbed }

            RSpec::Mocks.teardown
            RSpec::Mocks.setup

            object.singleton_class.send(:prepend, Module.new {
            })

            allow(object).to receive(:value) { :stubbed_2 }
            expect(object.value).to eq(:stubbed_2)
          end
        end
      end

      describe "#rspec_reset" do
        it "removes stubbed methods that didn't exist" do
          allow(@instance).to receive(:non_existent_method)
          reset @instance
          expect(@instance).not_to respond_to(:non_existent_method)
        end

        it "restores existing instance methods" do
          # See bug reports 8302 and 7805
          allow(@instance).to receive(:existing_instance_method) { :stub_value }
          reset @instance
          expect(@instance.existing_instance_method).to eq(:original_value)
        end

        it "restores existing singleton methods with the appropriate context" do
          klass = Class.new do
          end

          subclass = Class.new(klass)

          subclass.instance_variable_set(:@hello, "Hello")
          expect(subclass.say_hello).to eq("Hello")

          allow(klass).to receive(:say_hello) { "Howdy" }
          expect(subclass.say_hello).to eq("Howdy")

          reset klass
          expect(subclass.say_hello).to eq("Hello")
        end

        it "restores existing private instance methods" do
          # See bug reports 8302 and 7805
          allow(@instance).to receive(:existing_private_instance_method) { :stub_value }
          reset @instance
          expect(@instance.send(:existing_private_instance_method)).to eq(:original_value)
        end

        it "restores existing class methods" do
          # See bug reports 8302 and 7805
          allow(@class).to receive(:existing_class_method) { :stub_value }
          reset @class
          expect(@class.existing_class_method).to eq(:original_value)
        end

        it "restores existing aliased module_function methods" do
          m = Module.new do
      def initialize
        @full_backtrace = false

        patterns = %w[ /lib\d*/ruby/ bin/ exe/rspec /lib/bundler/ /exe/bundle: ]
        patterns << "org/jruby/" if RUBY_PLATFORM == 'java'
        patterns.map! { |s| Regexp.new(s.gsub("/", File::SEPARATOR)) }

        @exclusion_patterns = [Regexp.union(RSpec::CallerFilter::IGNORE_REGEX, *patterns)]
        @inclusion_patterns = []

        return unless matches?(@exclusion_patterns, File.join(Dir.getwd, "lib", "foo.rb:13"))
        inclusion_patterns << Regexp.new(Dir.getwd)
      end
            module_function :mkdir_p

            alias mkpath mkdir_p

            module_function :mkpath
          end

          allow(m).to receive(:mkpath) { :stub_value }
          allow(m).to receive(:mkdir_p) { :stub_value }
          reset m
          expect(m.mkpath).to eq(:mkdir_p)
          expect(m.mkdir_p).to eq(:mkdir_p)
        end

        it "restores existing private class methods" do
          # See bug reports 8302 and 7805
          allow(@class).to receive(:existing_private_class_method) { :stub_value }
          reset @class
          expect(@class.send(:existing_private_class_method)).to eq(:original_value)
        end

        it "does not remove existing methods that have been stubbed twice" do
          allow(@instance).to receive(:existing_instance_method)
          allow(@instance).to receive(:existing_instance_method)

          reset @instance

          expect(@instance.existing_instance_method).to eq(:original_value)
        end

        it "correctly restores the visibility of methods whose visibility has been tweaked on the singleton class" do
          # hello is a private method when mixed in, but public on the module
          # itself
          mod = Module.new {
            extend self
            def hello; :hello; end

            private :hello
            class << self; public :hello; end;
          }

          expect(mod.hello).to eq(:hello)

          allow(mod).to receive(:hello) { :stub }
          reset mod

          expect(mod.hello).to eq(:hello)
        end

        it "correctly handles stubbing inherited mixed in class methods" do
          mod = Module.new do
          def matches?(actual); actual; end
          def failure_message; "failure when positive"; end
          def negative_failure_message; "failure when negative"; end
        end

        it_behaves_like "a matcher written against a legacy protocol", matcher_class
      end
    end

      def handle_completed(completed)
        case completed
        when nil, # someone aborted the job but wants to call the on_complete callback
             true
          true
        when false
          false
        else
          raise "Unexpected thrown value: #{completed.inspect}"
        end
          end

          a = Class.new { include mod }
          b = Class.new(a) do
    def processes
      puts "---- Processes (#{process_set.size}) ----"
      process_set.each_with_index do |process, index|
        # Keep compatibility with legacy versions since we don't want to break sidekiqmon during rolling upgrades or downgrades.
        #
        # Before:
        #   ["default", "critical"]
        #
        # After:
        #   {"default" => 1, "critical" => 10}
        queues =
          if process["weights"]
            process["weights"].sort_by { |queue| queue[0] }.map { |capsule| capsule.map { |name, weight| (weight > 0) ? "#{name}: #{weight}" : name }.join(", ") }
          else
            process["queues"].sort
          end
          end

          allow(a).to receive(:method_a)
          allow(b).to receive(:method_b).and_return("stubbed method_b")

          expect(b.method_b).to eql("stubbed method_b")
        end

        if Support::RubyFeatures.module_prepends_supported?
          context "with a prepended module (ruby 2.0.0+)" do
            module ToBePrepended
      def initialize(hash) # :nodoc:
        @tempfile = hash[:tempfile]
        raise(ArgumentError, ":tempfile is required") unless @tempfile

        @content_type = hash[:type]

        if hash[:filename]
          @original_filename = hash[:filename].dup

          begin
            @original_filename.encode!(Encoding::UTF_8)
          rescue EncodingError
            @original_filename.force_encoding(Encoding::UTF_8)
          end
            end

            before do
              @prepended_class = Class.new do
                prepend ToBePrepended

      def register_type(key, value = nil, &block)
        raise ::ArgumentError unless value || block

        if block
          @mapping[key] = block
        else
          @mapping[key] = proc { value }
        end

              end
              @prepended_instance = @prepended_class.new
            end

            it "restores prepended instance methods" do
              allow(@prepended_instance).to receive(:existing_method) { :stubbed }
              expect(@prepended_instance.existing_method).to eq :stubbed

              reset @prepended_instance
              expect(@prepended_instance.existing_method).to eq :original_value_prepended
            end

            it "restores non-prepended instance methods" do
              allow(@prepended_instance).to receive(:non_prepended_method) { :stubbed }
              expect(@prepended_instance.non_prepended_method).to eq :stubbed

              reset @prepended_instance
              expect(@prepended_instance.non_prepended_method).to eq :not_prepended
            end

            it "restores prepended class methods" do
              klass = Class.new do
                class << self; prepend ToBePrepended; end
              end

              allow(klass).to receive(:existing_method) { :stubbed }
              expect(klass.existing_method).to eq :stubbed

              reset klass
              expect(klass.existing_method).to eq :original_value_prepended
            end

            it "restores prepended object singleton methods" do
              object = Object.new
        def with_keywords_string
          kw_str = case @expected_keywords.count
                   when 1
                     @expected_keywords.first.inspect
                   when 2
                     @expected_keywords.map(&:inspect).join(' and ')
                   else
                     "#{@expected_keywords[0...-1].map(&:inspect).join(', ')}, and #{@expected_keywords.last.inspect}"
                   end
          end
        end
      end

      it "returns values in order to consecutive calls" do
        allow(@instance).to receive(:msg).and_return("1", 2, :three)
        expect(@instance.msg).to eq("1")
        expect(@instance.msg).to eq(2)
        expect(@instance.msg).to eq(:three)
      end

      it "keeps returning last value in consecutive calls" do
        allow(@instance).to receive(:msg).and_return("1", 2, :three)
        expect(@instance.msg).to eq("1")
        expect(@instance.msg).to eq(2)
        expect(@instance.msg).to eq(:three)
        expect(@instance.msg).to eq(:three)
        expect(@instance.msg).to eq(:three)
      end

      it "yields a specified object" do
        allow(@instance).to receive(:method_that_yields).and_yield(:yielded_obj)
        current_value = :value_before
        @instance.method_that_yields { |val| current_value = val }
        expect(current_value).to eq :yielded_obj
        verify @instance
      end

      it "yields multiple times with multiple calls to and_yield" do
        allow(@instance).to receive(:method_that_yields_multiple_times).and_yield(:yielded_value).
                                                       and_yield(:another_value)
        current_value = []
        @instance.method_that_yields_multiple_times { |val| current_value << val }
        expect(current_value).to eq [:yielded_value, :another_value]
        verify @instance
      end

      it "yields a specified object and return another specified object" do
        yielded_obj = double("my mock")
        expect(yielded_obj).to receive(:foo).with(:bar)
        allow(@instance).to receive(:method_that_yields_and_returns).and_yield(yielded_obj).and_return(:baz)
        expect(@instance.method_that_yields_and_returns { |o| o.foo :bar }).to eq :baz
      end

      it "throws when told to" do
        allow(@stub).to receive(:something).and_throw(:up)
        expect { @stub.something }.to throw_symbol(:up)
      end

      it "throws with argument when told to" do
        allow(@stub).to receive(:something).and_throw(:up, 'high')
        expect { @stub.something }.to throw_symbol(:up, 'high')
      end

      it "overrides a pre-existing method" do
        allow(@stub).to receive(:existing_instance_method).and_return(:updated_stub_value)
        expect(@stub.existing_instance_method).to eq :updated_stub_value
      end

      it "overrides a pre-existing stub" do
        allow(@stub).to receive(:foo) { 'bar' }
        allow(@stub).to receive(:foo) { 'baz' }
        expect(@stub.foo).to eq 'baz'
      end

      it "allows a stub and an expectation" do
        allow(@stub).to receive(:foo).with("bar")
        expect(@stub).to receive(:foo).with("baz")
        @stub.foo("bar")
        @stub.foo("baz")
      end
    end

    RSpec.describe "A method stub with args" do
      before(:each) do
        @stub = Object.new
        allow(@stub).to receive(:foo).with("bar")
      end

      it "does not complain if not called" do
      end

      it "does not complain if called with arg" do
        @stub.foo("bar")
      end

      it "complains if called with no arg" do
        expect {
          @stub.foo
        }.to raise_error(/received :foo with unexpected arguments/)
      end

      it "complains if called with other arg", :github_issue => [123, 147] do
        expect {
          @stub.foo("other")
        }.to raise_error(/received :foo with unexpected arguments.*Please stub a default value/m)
      end

      it "does not complain if also mocked w/ different args" do
        expect(@stub).to receive(:foo).with("baz")
        @stub.foo("bar")
        @stub.foo("baz")
      end

      it "complains if also mocked w/ different args AND called w/ a 3rd set of args" do
        expect(@stub).to receive(:foo).with("baz")
        @stub.foo("bar")
        @stub.foo("baz")
        expect {
          @stub.foo("other")
        }.to fail
      end

      it 'uses the correct stubbed response when responding to a mock expectation' do
        allow(@stub).to receive(:bar) { 15 }
        allow(@stub).to receive(:bar).with(:eighteen) { 18 }
        allow(@stub).to receive(:bar).with(:thirteen) { 13 }

        expect(@stub).to receive(:bar).exactly(4).times

        expect(@stub.bar(:blah)).to eq(15)
        expect(@stub.bar(:thirteen)).to eq(13)
        expect(@stub.bar(:eighteen)).to eq(18)
        expect(@stub.bar).to eq(15)
      end
    end
  end
end
