function [ number ] = generateRandom(range)
  %generateRandom Generates a random number.
  number = rand;
  number = number*range;
end