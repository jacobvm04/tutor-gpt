import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { useState } from "react";
import { useRouter } from "next/navigation";

import Swal from 'sweetalert2'

export default function SignUp(props: any) {
  const { stateSync } = props
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirmation, setPasswordConfirmation] = useState('');
  const [opt, setOpt] = useState<boolean>(true)
  const router = useRouter();
  const supabase = createClientComponentClient()

  const handleSignUp = async (e: any) => {
    e.preventDefault();
    if (password !== passwordConfirmation) {
      await Swal.fire({
        title: "Passwords don't match",
        icon: 'error',
        text: 'Re-confirm you password and try again',
      })
      return
    }
    const { error } = await supabase.auth.signUp(
      {
        email,
        password,
        options: {
          emailRedirectTo: `${location.origin}/`,
          data: {
            dataOptIn: opt
          }
        }
      });
    if (error) {
      console.error(error);
    } else {
      Swal.fire({
        title: "Success",
        icon: "success",
        text: "Please check your email for a verification link"
      })
    }
  };

  return (
    <form action="#" className="mt-8 grid grid-cols-6 gap-6">

      <div className="col-span-6">
        <label htmlFor="Email" className="block text-sm font-medium text-gray-700">
          Email
        </label>

        <input
          type="email"
          id="Email"
          name="email"
          className="p-2 mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
          value={email} onChange={(e) => setEmail(e.target.value)}
        />
      </div>

      <div className="col-span-6 sm:col-span-3">
        <label
          htmlFor="Password"
          className="block text-sm font-medium text-gray-700"
        >
          Password
        </label>

        <input
          type="password"
          id="Password"
          name="password"
          className="p-2 mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
          value={password} onChange={(e) => setPassword(e.target.value)}
        />
      </div>

      <div className="col-span-6 sm:col-span-3">
        <label
          htmlFor="PasswordConfirmation"
          className="block text-sm font-medium text-gray-700"
        >
          Password Confirmation
        </label>

        <input
          type="password"
          id="PasswordConfirmation"
          name="password_confirmation"
          className="p-2 mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
          value={passwordConfirmation} onChange={(e) => setPasswordConfirmation(e.target.value)}
        />
      </div>

      <div className="col-span-6">
        <label htmlFor="MarketingAccept" className="flex gap-4">
          <input
            type="checkbox"
            id="MarketingAccept"
            name="marketing_accept"
            className="h-5 w-5 rounded-md border-gray-200 bg-white shadow-sm"
            checked={opt}
            onChange={(e) => setOpt(!opt)}
          />

          <span className="text-sm text-gray-700">
            I want to receive emails about events, product updates and
            company announcements.
          </span>
        </label>
      </div>

      <div className="col-span-6">
        <p className="text-sm text-gray-500">
          By creating an account, you agree to our{' '}
          <a href="https://app.termly.io/document/terms-of-service/ba5ac452-fdd6-4746-8b31-973351d05008" target="_blank" className="text-gray-700 underline">
            terms and conditions
          </a>
          {' '}and{' '}
          <a href="https://app.termly.io/document/privacy-policy/29672110-b634-40ae-854d-ebaf55e8fa75" target="_blank" className="text-gray-700 underline">privacy policy</a>.
        </p>
      </div>

      <div className="col-span-6 sm:flex sm:items-center sm:gap-4">
        <button
          className="inline-block shrink-0 rounded-md border border-neon-green bg-neon-green px-12 py-3 text-sm font-medium transition hover:bg-transparent hover:text-blue-600 focus:outline-none focus:ring active:text-blue-500"
          onClick={handleSignUp}
        >
          Create an account
        </button>

        <p className="mt-4 text-sm text-gray-500 sm:mt-0">
          Already have an account?{' '}
          <a href="#" onClick={() => { console.log("help"); stateSync("LOGIN"); }} className="text-gray-700 underline">Log in</a>.
        </p>
      </div>
    </form>
  )
}
