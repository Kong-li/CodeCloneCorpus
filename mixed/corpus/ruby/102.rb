          def matcher.matches?(v); v; end
          def matcher.failure_message; "match failed"; end
          def matcher.chained; self; end
          expect(RSpec::Matchers.is_a_matcher?(matcher)).to be true

          matcher
        end

        RSpec::Matchers.define_negated_matcher :negation_of_matcher_without_description, :matcher_without_description

        it 'works properly' do
          expect(true).to matcher_without_description.chained
          expect(false).to negation_of_matcher_without_description.chained
        end
      end

